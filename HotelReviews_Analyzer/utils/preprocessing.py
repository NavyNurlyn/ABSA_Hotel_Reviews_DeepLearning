"""
utils/preprocessing.py — VERSI FIXED
=====================================
Perbaikan:
1. Kamus alay: URL okkyibrohim + format CSV no-header (2 kolom: slang,formal)
2. Tambah normalisasi 'staff' → 'staf' dan kata domain lain
3. Tambah tampilkan hasil translate dan confidence
4. Fix warning TF retracing
"""

import re
import pickle
import requests
import numpy as np
from pathlib import Path

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

try:
    import indoNLP.preprocessing as indoprep
    INLP_OK = True
except Exception:
    INLP_OK = False

try:
    from langdetect import detect, LangDetectException
    LANG_OK = True
except Exception:
    LANG_OK = False

try:
    from deep_translator import GoogleTranslator
    TRANS_OK = True
except Exception:
    TRANS_OK = False


# ════════════════════════════════════════════
# KAMUS ALAY — URL YANG BENAR
# Format: no header, 2 kolom (slang,formal)
# ════════════════════════════════════════════
KAMUS_ALAY_URL = (
    "https://raw.githubusercontent.com/okkyibrohim/"
    "id-multi-label-hate-speech-and-abusive-language-detection"
    "/master/new_kamusalay.csv"
)
_kamus_alay_dict: dict = {}


def _load_kamus_alay() -> dict:
    global _kamus_alay_dict
    if _kamus_alay_dict:
        return _kamus_alay_dict
    try:
        import pandas as pd
        import io
        resp = requests.get(KAMUS_ALAY_URL, timeout=15)
        resp.raise_for_status()

        # Format: NO header, 2 kolom (slang,formal)
        df = pd.read_csv(
            io.StringIO(resp.text),
            encoding='utf-8',
            header=None,           # tidak ada header!
            names=['slang','formal'],
            on_bad_lines='skip'    # skip baris bermasalah
        )
        # Pastikan hanya 2 kolom
        df = df[['slang','formal']].dropna()
        _kamus_alay_dict = dict(zip(
            df['slang'].astype(str).str.lower().str.strip(),
            df['formal'].astype(str).str.lower().str.strip()
        ))
        print(f"[Kamus Alay] Loaded: {len(_kamus_alay_dict):,} entri")
    except Exception as e:
        print(f"[WARNING] Kamus alay gagal dimuat: {e}")
        _kamus_alay_dict = {}
    return _kamus_alay_dict


# ════════════════════════════════════════════
# FRASA DOMAIN HOTEL
# ════════════════════════════════════════════
FRASA_HOTEL = {
    'check in'        : 'checkin',
    'check out'       : 'checkout',
    'cek in'          : 'checkin',
    'cek out'         : 'checkout',
    'early check in'  : 'earlycheckin',
    'late check out'  : 'latecheckout',
    'air conditioner' : 'ac',
    'air conditioning': 'ac',
    'wi fi'           : 'wifi',
    'worth it'        : 'worthit',
}

# ════════════════════════════════════════════
# CUSTOM NORMALIZATION DICT
# Tambahan: staff → staf, dll
# ════════════════════════════════════════════
CUSTOM_NORM_DICT = {
    # PERBAIKAN BUG UTAMA: negasi
    # Jalankan SEBELUM kamus alay agar nggak → tidak
    # bukan nggak → enggak (via kamus alay) → hilang di stopword
    'nggak'     : 'tidak',
    'nggk'      : 'tidak',
    'ngga'      : 'tidak',
    'gak'       : 'tidak',
    'ga'        : 'tidak',
    'enggak'    : 'tidak',
    'ngak'      : 'tidak',
    'kagak'     : 'tidak',
    'tdk'       : 'tidak',
    'gaada'     : 'tidak ada',
    'gada'      : 'tidak ada',

    # Kata domain hotel — normalisasi ejaan
    'staff'     : 'staf',      # staff (Inggris) → staf (Indonesia)
    'staffnya'  : 'stafnya',
    'staffs'    : 'staf',
    'resepsionis': 'resepsionis',
    'receptionist': 'resepsionis',
    'lobby'     : 'lobi',
    'lift'      : 'lift',
    'elevator'  : 'lift',
    'pool'      : 'kolam renang',
    'breakfast' : 'sarapan',
    'buffet'    : 'prasmanan',
    'balcony'   : 'balkon',
    'parking'   : 'parkir',
    'laundry'   : 'laundri',

    # Koreksi hasil kamus alay yang tidak sesuai konteks
    'plg'       : 'paling',
    'cuman'     : 'cuma',
    'dicuekin'  : 'dicueki',

    # Kata tidak ada di kamus alay
    'shngga'    : 'sehingga',
    'gue'       : 'saya',
    'knpnya'    : 'kenapa',
    'kyk'       : 'seperti',
    'kayak'     : 'seperti',
    'ky'        : 'seperti',
    'lelet'     : 'lambat',
    'sendal'    : 'sandal',
    'batrai'    : 'baterai',
    'remot'     : 'remote',
    'trimakasi' : 'terima kasih',
    'berasa'    : 'terasa',
    'kesumbat'  : 'tersumbat',
    'nginap'    : 'menginap',
}

# ════════════════════════════════════════════
# STOPWORD
# ════════════════════════════════════════════
_sw_final: set = set()


def _build_stopword() -> set:
    global _sw_final
    if _sw_final:
        return _sw_final

    sw_factory = StopWordRemoverFactory()
    stopwords  = set(sw_factory.get_stop_words())

    kata_wajib = {
        'tidak', 'tak', 'bukan', 'bukanlah', 'bukannya', 'bukankah',
        'belum', 'jangan', 'jangankan', 'tanpa',
        'sangat', 'amat', 'amatlah', 'sangatlah',
        'lebih', 'paling', 'kurang', 'cukup', 'terlalu',
        'agak', 'sekali', 'berlebihan', 'keterlaluan',
        'sedikit', 'sedikitnya', 'hampir', 'nyaris',
        'luar', 'biasa', 'banyak',
        'padahal', 'namun', 'tetapi', 'tapi',
        'meski', 'meskipun', 'walau', 'walaupun',
        'malah', 'malahan', 'justru', 'bahkan', 'sebaliknya',
        'memang', 'ternyata', 'sebenarnya', 'sebetulnya',
        'seharusnya', 'masih', 'sudah', 'pernah',
        'selalu', 'sering', 'sempat',
        'besar', 'kecil', 'enak',
        'dekat', 'jauh', 'tepat', 'berada',
        'sekitarnya', 'belakang', 'depan', 'tempat',
        'lama',
        'baik', 'benar', 'jelas', 'pantas', 'percuma',
        'penting', 'pasti', 'kembali', 'perlu',
        'mau', 'bisa', 'dapat', 'terasa', 'terlihat',
    }

    custom_tambahan = {
        'juga', 'sih', 'deh', 'lho', 'nih', 'lah',
        'kan', 'wow', 'nya', 'kok', 'dong', 'oh',
    }

    sw_tahap1  = stopwords - kata_wajib
    _sw_final  = sw_tahap1 | custom_tambahan
    return _sw_final


# ════════════════════════════════════════════
# STEMMER
# ════════════════════════════════════════════
_stemmer = None


def _get_stemmer():
    global _stemmer
    if _stemmer is None:
        _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


# ════════════════════════════════════════════
# FUNGSI PREPROCESSING
# ════════════════════════════════════════════

def text_cleaning(text: str) -> str:
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _normalize_frasa(text: str) -> str:
    for frasa, ganti in sorted(
        FRASA_HOTEL.items(), key=lambda x: len(x[0]), reverse=True
    ):
        text = re.sub(r'\b' + re.escape(frasa) + r'\b', ganti, text)
    return text


def _elongasi(text: str) -> str:
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def normalize_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = _normalize_frasa(text)
    text = _elongasi(text)

    # Custom dict DULU (penting: nggak → tidak sebelum kamus alay)
    tokens = text.split()
    tokens = [CUSTOM_NORM_DICT.get(t, t) for t in tokens]
    text   = ' '.join(tokens)

    # Kamus alay sesudah custom dict
    kamus = _load_kamus_alay()
    if kamus:
        tokens = text.split()
        tokens = [kamus.get(t, t) for t in tokens]
        text   = ' '.join(tokens)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text: str) -> str:
    if not isinstance(text, str) or text.strip() == '':
        return ''
    sw     = _build_stopword()
    tokens = [t for t in text.split() if t not in sw]
    return ' '.join(tokens)


def stem_text(text: str) -> str:
    if not isinstance(text, str) or text.strip() == '':
        return ''
    stemmer = _get_stemmer()
    tokens  = [stemmer.stem(t) for t in text.split()]
    return ' '.join(tokens)


def full_preprocessing(raw_text: str):
    """
    Input : teks mentah (sudah dalam bahasa Indonesia)
    Output: (review_non_stemmed, review_stemmed)
    """
    text = text_cleaning(raw_text)
    if not text:
        return '', ''
    text    = text.lower()
    text    = normalize_text(text)
    text    = remove_stopwords(text)
    nonstem = text
    stemmed = stem_text(text)
    return nonstem, stemmed


# ════════════════════════════════════════════
# DETEKSI BAHASA & TRANSLATE
# ════════════════════════════════════════════
SUPPORTED_LANGS = {'id', 'en'}


def detect_language(text: str) -> str:
    if not LANG_OK:
        return 'id'
    try:
        return detect(str(text))
    except Exception:
        return 'unknown'


def translate_to_indonesian(text: str, lang: str) -> str:
    if lang == 'id':
        return text
    if not TRANS_OK:
        return text
    try:
        result = GoogleTranslator(source=lang, target='id').translate(text)
        return result if result else text
    except Exception:
        return text


def process_input(raw_text: str) -> dict:
    """
    Pipeline lengkap dari teks mentah.

    Return dict:
    {
      'lang'        : kode bahasa
      'supported'   : True/False
      'translated'  : teks setelah translate (sama jika sudah Indo)
      'nonstem'     : review_non_stemmed
      'stemmed'     : review_stemmed
      'too_short'   : True jika < 3 kata
      'is_translated': True jika benar-benar ditranslate
    }
    """
    lang = detect_language(raw_text)

    if lang not in SUPPORTED_LANGS and lang != 'unknown':
        return {
            'lang'          : lang,
            'supported'     : False,
            'translated'    : raw_text,
            'nonstem'       : '',
            'stemmed'       : '',
            'too_short'     : False,
            'is_translated' : False,
        }

    is_translated = (lang == 'en')
    translated    = translate_to_indonesian(raw_text, lang)
    nonstem, stemmed = full_preprocessing(translated)
    too_short     = len(nonstem.split()) < 3

    return {
        'lang'          : lang,
        'supported'     : True,
        'translated'    : translated,
        'nonstem'       : nonstem,
        'stemmed'       : stemmed,
        'too_short'     : too_short,
        'is_translated' : is_translated,
    }
