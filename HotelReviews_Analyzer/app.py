import os
import io
import json
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np
from flask import (
    Flask, render_template, request,
    jsonify, send_file, url_for
)
from werkzeug.utils import secure_filename

from utils.preprocessing import process_input
from utils.pipeline import get_pipeline

BASE_DIR      = Path(__file__).parent
MODEL_DIR     = BASE_DIR / 'models'
TOKENIZER_DIR = BASE_DIR / 'tokenizers'
UPLOAD_FOLDER = BASE_DIR / 'uploads_temp'
UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
MAX_FILE_MB        = 10

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_MB * 1024 * 1024
app.config['SECRET_KEY'] = 'hotel-analyzer-secret-2024'

try:
    pipeline       = get_pipeline(str(MODEL_DIR), str(TOKENIZER_DIR))
    PIPELINE_READY = True
    print("[APP] Pipeline ABSA siap!")
except Exception as e:
    PIPELINE_READY = False
    pipeline       = None
    print(f"[APP] WARNING: Pipeline gagal load: {e}")


def allowed_file(filename: str) -> bool:
    return ('.' in filename and
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)


def analisis_satu_ulasan(teks_raw: str) -> dict:
    t0   = time.time()
    prep = process_input(teks_raw)

    if not prep['supported']:
        return {
            'status': 'error',
            'pesan' : (f"Bahasa tidak didukung ({prep['lang']}). "
                       "Sistem hanya mendukung Bahasa Indonesia dan Inggris."),
            'lang'  : prep['lang'],
        }

    if prep['too_short']:
        return {
            'status': 'error',
            'pesan' : "Ulasan terlalu pendek (kurang dari 3 kata setelah preprocessing).",
            'lang'  : prep['lang'],
        }

    # Prediksi dengan confidence score
    hasil, confidence = pipeline.predict_single_with_confidence(
        prep['nonstem'], prep['stemmed']
    )

    elapsed = round(time.time() - t0, 3)

    return {
        'status'        : 'ok',
        'lang'          : prep['lang'],
        'is_translated' : prep['is_translated'],
        'translated'    : prep['translated'],    # teks hasil translate
        'teks_asli'     : teks_raw,
        'nonstem'       : prep['nonstem'],
        'stemmed'       : prep['stemmed'],
        'hasil'         : hasil,
        'confidence'    : confidence,            # dict {'Room': 0.92, ...}
        'waktu_detik'   : elapsed,
    }


def buat_keyword_per_aspek(df_hasil: pd.DataFrame,
                            aspek: str, sentimen: str,
                            top_n: int = 15) -> list:
    """
    PERBAIKAN: kolom di df_hasil adalah 'Aspek Room', 'Aspek Hotel', dst
    bukan 'Room', 'Hotel'
    """
    from collections import Counter

    # Nama kolom di df_out adalah 'Aspek Room', bukan 'Room'
    col_aspek = f'Aspek {aspek}'

    if col_aspek not in df_hasil.columns:
        return []

    mask = df_hasil[col_aspek] == sentimen
    if mask.sum() == 0:
        return []

    df_filtered = df_hasil[mask]

    # Pakai kolom nonstem untuk keyword yang lebih natural
    if 'nonstem' not in df_filtered.columns:
        return []

    semua_teks = ' '.join(
        df_filtered['nonstem'].fillna('').astype(str).tolist()
    )

    counter = Counter(semua_teks.split())

    # Filter kata terlalu umum
    kata_umum = {
        'hotel', 'kamar', 'lokasi', 'pelayanan', 'service',
        'room', 'location', 'place', 'yang', 'dan', 'di',
        'ke', 'ini', 'itu', 'dengan', 'untuk', 'ada', 'dari',
        'tidak', 'sangat', 'juga', 'sudah', 'lebih', 'sangat',
        'bagus', 'baik', 'oke', 'ok',
    }
    for k in kata_umum:
        counter.pop(k, None)

    return [{'kata': k, 'frekuensi': v}
            for k, v in counter.most_common(top_n)
            if len(k) > 2]


# ══════════════════════════════════════
# ROUTES
# ══════════════════════════════════════

@app.route('/')
def home():
    return render_template('home.html', pipeline_ready=PIPELINE_READY)


@app.route('/analisis-teks')
def analisis_teks():
    return render_template('analisis_teks.html', pipeline_ready=PIPELINE_READY)


@app.route('/analisis-file')
def analisis_file():
    return render_template('analisis_file.html', pipeline_ready=PIPELINE_READY)


@app.route('/api/analisis-teks', methods=['POST'])
def api_analisis_teks():
    if not PIPELINE_READY:
        return jsonify({'status': 'error', 'pesan': 'Model belum siap.'}), 503

    data = request.get_json()
    if not data or 'teks' not in data:
        return jsonify({'status': 'error', 'pesan': 'Field "teks" wajib diisi.'}), 400

    teks = str(data['teks']).strip()
    if not teks:
        return jsonify({'status': 'error', 'pesan': 'Teks kosong.'}), 400

    try:
        hasil = analisis_satu_ulasan(teks)
        return jsonify(hasil)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'pesan': str(e)}), 500


@app.route('/api/analisis-file', methods=['POST'])
def api_analisis_file():
    if not PIPELINE_READY:
        return jsonify({'status': 'error', 'pesan': 'Model belum siap.'}), 503

    if 'file' not in request.files:
        return jsonify({'status': 'error', 'pesan': 'File tidak ditemukan.'}), 400

    f = request.files['file']
    if f.filename == '' or not allowed_file(f.filename):
        return jsonify({'status': 'error',
                        'pesan': 'Format tidak valid. Gunakan .csv, .xlsx, .xls'}), 400

    try:
        fname = secure_filename(f.filename)
        if fname.endswith('.csv'):
            df = pd.read_csv(f, keep_default_na=False)
        else:
            df = pd.read_excel(f, keep_default_na=False)

        if 'Teks Ulasan' not in df.columns:
            return jsonify({
                'status': 'error',
                'pesan' : 'Kolom "Teks Ulasan" tidak ditemukan. Download template.'
            }), 400

        df = df.fillna('')
        df['Teks Ulasan'] = df['Teks Ulasan'].astype(str).str.strip()
        df = df[df['Teks Ulasan'] != ''].reset_index(drop=True)

        if len(df) == 0:
            return jsonify({'status': 'error', 'pesan': 'File tidak berisi ulasan.'}), 400
        if len(df) > 1000:
            return jsonify({'status': 'error',
                            'pesan': f'Maksimum 1000 ulasan. File kamu: {len(df)} baris.'}), 400

        # Preprocessing semua ulasan
        prep_results = []
        skipped      = []

        for idx, row in df.iterrows():
            teks = row['Teks Ulasan']
            prep = process_input(teks)

            if not prep['supported']:
                skipped.append({'idx': idx, 'lang': prep['lang'], 'teks': teks[:50]})
                prep_results.append(None)
            elif prep['too_short']:
                prep_results.append(None)
            else:
                prep_results.append(prep)

        # Batch prediksi
        valid_indices = [i for i, p in enumerate(prep_results) if p is not None]
        valid_preps   = [prep_results[i] for i in valid_indices]

        # Inisialisasi hasil default semua '-'
        batch_results = [
            {'Room': '-', 'Hotel': '-', 'Location': '-', 'Service': '-'}
            for _ in range(len(df))
        ]

        if valid_preps:
            preds = pipeline.predict_batch(valid_preps)
            for i, pred in zip(valid_indices, preds):
                batch_results[i] = pred

        # Susun output DataFrame
        df_out = df.copy()
        # PENTING: nama kolom pakai prefix 'Aspek ' agar konsisten
        for aspek in ['Room', 'Hotel', 'Location', 'Service']:
            df_out[f'Aspek {aspek}'] = [r[aspek] for r in batch_results]

        # Simpan kolom preprocessing untuk keyword extraction
        df_out['lang']    = [p['lang']    if p else 'unknown' for p in prep_results]
        df_out['nonstem'] = [p['nonstem'] if p else ''        for p in prep_results]
        df_out['stemmed'] = [p['stemmed'] if p else ''        for p in prep_results]

        # Kolom yang ditampilkan di tabel frontend
        kolom_tampil = ['Teks Ulasan', 'Tanggal Ulasan', 'Rating',
                        'Username', 'Aspek Room', 'Aspek Hotel',
                        'Aspek Location', 'Aspek Service']
        kolom_ada    = [c for c in kolom_tampil if c in df_out.columns]
        tabel        = df_out[kolom_ada].to_dict('records')

        # Statistik dashboard
        stats = _hitung_statistik(df_out)

        return jsonify({
            'status'  : 'ok',
            'n_total' : len(df),
            'n_valid' : len(valid_preps),
            'n_skip'  : len(skipped),
            'skipped' : skipped[:5],
            'tabel'   : tabel,
            'stats'   : stats,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'pesan': f'Error: {str(e)}'}), 500


def _hitung_statistik(df_out: pd.DataFrame) -> dict:
    """Hitung statistik untuk dashboard. Kolom pakai prefix 'Aspek '."""
    aspek_cols  = ['Aspek Room', 'Aspek Hotel', 'Aspek Location', 'Aspek Service']
    aspek_names = ['Room', 'Hotel', 'Location', 'Service']

    # Distribusi aspek
    aspek_dist = {}
    for col, name in zip(aspek_cols, aspek_names):
        if col in df_out.columns:
            aspek_dist[name] = int((df_out[col] != '-').sum())

    # Sentimen per aspek
    sentimen_per_aspek = {}
    for col, name in zip(aspek_cols, aspek_names):
        if col not in df_out.columns:
            continue
        sentimen_per_aspek[name] = {
            'Positif': int((df_out[col] == 'Positif').sum()),
            'Negatif': int((df_out[col] == 'Negatif').sum()),
        }

    total_pos = sum(int((df_out[c] == 'Positif').sum())
                    for c in aspek_cols if c in df_out.columns)
    total_neg = sum(int((df_out[c] == 'Negatif').sum())
                    for c in aspek_cols if c in df_out.columns)

    # Time series per bulan
    time_series = []
    if 'Tanggal Ulasan' in df_out.columns:
        try:
            df_ts = df_out.copy()
            df_ts['tgl']   = pd.to_datetime(df_ts['Tanggal Ulasan'], errors='coerce')
            df_ts          = df_ts.dropna(subset=['tgl'])
            df_ts['bulan'] = df_ts['tgl'].dt.to_period('M').astype(str)

            for col, name in zip(aspek_cols, aspek_names):
                if col not in df_ts.columns:
                    continue
                for bulan, grp in df_ts.groupby('bulan'):
                    pos = int((grp[col] == 'Positif').sum())
                    neg = int((grp[col] == 'Negatif').sum())
                    if pos + neg > 0:
                        time_series.append({
                            'aspek': name, 'bulan': bulan,
                            'Positif': pos, 'Negatif': neg,
                        })
        except Exception:
            pass

    # Keyword per aspek
    keywords = {}
    for name in aspek_names:
        keywords[name] = {
            'Positif': buat_keyword_per_aspek(df_out, name, 'Positif', 15),
            'Negatif': buat_keyword_per_aspek(df_out, name, 'Negatif', 15),
        }

    return {
        'aspek_dist'        : aspek_dist,
        'sentimen_per_aspek': sentimen_per_aspek,
        'total_pos'         : total_pos,
        'total_neg'         : total_neg,
        'time_series'       : time_series,
        'keywords'          : keywords,
    }


@app.route('/api/download', methods=['POST'])
def api_download():
    data = request.get_json()
    if not data or 'tabel' not in data:
        return jsonify({'status': 'error', 'pesan': 'Data tidak valid.'}), 400

    try:
        df  = pd.DataFrame(data['tabel'])
        fmt = data.get('format', 'csv')

        if fmt == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False, encoding='utf-8-sig')
            output.seek(0)
            return send_file(
                io.BytesIO(output.read().encode('utf-8-sig')),
                mimetype='text/csv',
                as_attachment=True,
                download_name='hasil_analisis_hotel.csv'
            )
        else:
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Hasil Analisis')
                wb  = writer.book
                ws  = writer.sheets['Hasil Analisis']
                fmt_pos = wb.add_format({'bg_color': '#d4edda', 'font_color': '#155724'})
                fmt_neg = wb.add_format({'bg_color': '#f8d7da', 'font_color': '#721c24'})
                fmt_non = wb.add_format({'bg_color': '#e2e3e5', 'font_color': '#383d41'})
                aspek_cols = ['Aspek Room', 'Aspek Hotel', 'Aspek Location', 'Aspek Service']
                for col_name in aspek_cols:
                    if col_name not in df.columns:
                        continue
                    col_idx = df.columns.get_loc(col_name)
                    for row_idx, val in enumerate(df[col_name], start=1):
                        fmt_cell = (fmt_pos if val == 'Positif'
                                    else fmt_neg if val == 'Negatif'
                                    else fmt_non)
                        ws.write(row_idx, col_idx, val, fmt_cell)
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='hasil_analisis_hotel.xlsx'
            )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'pesan': str(e)}), 500


@app.route('/api/template')
def api_template():
    template_data = {
        'Teks Ulasan'   : [
            'Kamarnya bersih dan nyaman, AC dingin, view bagus sekali!',
            'Staff sangat ramah dan membantu, pelayanan sangat baik',
            'Lokasi strategis dekat pusat kota dan mall',
            'The room was dirty and small, not worth the price',
        ],
        'Tanggal Ulasan': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05'],
        'Rating'        : [5, 4, 5, 2],
        'Username'      : ['user1', 'user2', 'user3', 'user4'],
    }
    df  = pd.DataFrame(template_data)
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Template')
        wb     = writer.book
        ws     = writer.sheets['Template']
        hdr_fmt = wb.add_format({
            'bold': True, 'bg_color': '#1a56db',
            'font_color': 'white', 'border': 1
        })
        for col_idx, col_name in enumerate(df.columns):
            ws.write(0, col_idx, col_name, hdr_fmt)
            ws.set_column(col_idx, col_idx, 30)
    out.seek(0)
    return send_file(
        out,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        as_attachment=True,
        download_name='template_analisis_hotel.xlsx'
    )


if __name__ == '__main__':
    print("\n" + "="*55)
    print("  HotelAnalyzer AI")
    print("="*55)
    print(f"  Pipeline ready: {PIPELINE_READY}")
    print("="*55 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
