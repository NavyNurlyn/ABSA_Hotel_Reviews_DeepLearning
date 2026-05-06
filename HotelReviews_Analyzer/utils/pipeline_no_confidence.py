"""
utils/pipeline.py
=================
Pipeline prediksi ABSA 5-task.
Versi: TF 2.16 + Keras 3 (kompatibel dengan model Colab TF 2.19)

Skenario terbaik per task:
  Task 1 Aspek    : BiLSTM + FastText  + STEM     → tokenizer_stem
  Task 2 Room     : CNN    + FastText  + STEM     → tokenizer_stem
  Task 3 Hotel    : BiLSTM + Word2Vec  + NON-STEM → tokenizer_nonstem
  Task 4 Location : CNN    + Word2Vec  + NON-STEM → tokenizer_nonstem
  Task 5 Service  : CNN    + Word2Vec  + STEM     → tokenizer_stem
"""

import os
import pickle
import time
import numpy as np
from pathlib import Path

# Matikan warning TF yang tidak perlu
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# TF 2.16 pakai Keras 3 — import dari keras langsung
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    # Fallback untuk Keras 3
    from keras.src.legacy.preprocessing.sequence import pad_sequences

MAX_LENGTH  = 100
THRESHOLD   = 0.5
ASPEK_NAMES = ['Room', 'Hotel', 'Location', 'Service']

# Tokenizer yang dipakai setiap task
# PENTING: sesuai dengan skenario terbaik!
TASK_CONFIG = {
    'task1': 'stem',     # BiLSTM + FastText + STEM
    'task2': 'stem',     # CNN + FastText + STEM
    'task3': 'nonstem',  # BiLSTM + Word2Vec + NON-STEM
    'task4': 'nonstem',  # CNN + Word2Vec + NON-STEM
    'task5': 'stem',     # CNN + Word2Vec + STEM
}


class ABSAPipeline:

    def __init__(self, model_dir: str, tokenizer_dir: str):
        self.model_dir     = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.models        = {}
        self.tokenizers    = {}
        self._loaded       = False

    def load(self):
        print("[Pipeline] Loading tokenizer dan model...")
        t0 = time.time()

        # ── Load tokenizer ──
        for name in ['tokenizer_stem', 'tokenizer_nonstem']:
            path = self.tokenizer_dir / f'{name}.pkl'
            if not path.exists():
                raise FileNotFoundError(
                    f"Tokenizer tidak ditemukan: {path}\n"
                    f"Pastikan {name}.pkl ada di folder tokenizers/"
                )
            with open(path, 'rb') as f:
                self.tokenizers[name] = pickle.load(f)
            vocab = len(self.tokenizers[name].word_index) + 1
            print(f"  OK {name}: vocab={vocab:,}")

        # ── Load model ──
        model_files = {
            'task1': 'task1_aspek.h5',
            'task2': 'task2_room.h5',
            'task3': 'task3_hotel.h5',
            'task4': 'task4_location.h5',
            'task5': 'task5_service.h5',
        }

        for task, fname in model_files.items():
            path = self.model_dir / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Model tidak ditemukan: {path}\n"
                    f"Pastikan file {fname} ada di folder models/"
                )

            print(f"  Loading {fname}...")
            # TF 2.16 + Keras 3 langsung bisa baca .h5 dari Colab TF 2.19
            self.models[task] = tf.keras.models.load_model(
                str(path),
                compile=False
            )
            print(f"  OK {task}: {fname}")

        self._loaded = True
        print(f"[Pipeline] Semua siap dalam {time.time()-t0:.1f}s")

    def _tokenize(self, text: str, task: str) -> np.ndarray:
        """Tokenize dan pad teks sesuai task."""
        text_type = TASK_CONFIG[task]
        tokenizer = self.tokenizers[f'tokenizer_{text_type}']
        seq = tokenizer.texts_to_sequences([text])
        return pad_sequences(
            seq, maxlen=MAX_LENGTH,
            padding='post', truncating='post'
        )

    def predict_single(self, nonstem: str, stemmed: str) -> dict:
        """Prediksi ABSA untuk 1 ulasan."""
        if not self._loaded:
            raise RuntimeError("Pipeline belum di-load!")

        result = {a: '-' for a in ASPEK_NAMES}

        # Task 1: deteksi aspek (input: stemmed)
        X1    = self._tokenize(stemmed, 'task1')
        prob1 = self.models['task1'].predict(X1, verbose=0)[0]
        aspek = (prob1 >= THRESHOLD).astype(int)

        # Task 2-5: sentimen per aspek
        # Hanya jalankan jika aspek terdeteksi di Task 1
        task_map = {
            0: ('task2', stemmed),    # Room     → stem
            1: ('task3', nonstem),    # Hotel    → nonstem
            2: ('task4', nonstem),    # Location → nonstem
            3: ('task5', stemmed),    # Service  → stem
        }
        for idx, (task_key, teks) in task_map.items():
            if aspek[idx] == 1:
                X    = self._tokenize(teks, task_key)
                prob = self.models[task_key].predict(X, verbose=0)[0][0]
                result[ASPEK_NAMES[idx]] = (
                    'Positif' if prob >= THRESHOLD else 'Negatif'
                )

        return result

    def predict_batch(self, rows: list) -> list:
        """Prediksi batch untuk banyak ulasan sekaligus."""
        if not self._loaded:
            raise RuntimeError("Pipeline belum di-load!")
        if not rows:
            return []

        stemmed_list = [r['stemmed']  for r in rows]
        nonstem_list = [r['nonstem']  for r in rows]

        def batch_tok(texts, task):
            text_type = TASK_CONFIG[task]
            tok = self.tokenizers[f'tokenizer_{text_type}']
            seqs = tok.texts_to_sequences(texts)
            return pad_sequences(
                seqs, maxlen=MAX_LENGTH,
                padding='post', truncating='post'
            )

        # Task 1: deteksi aspek semua ulasan
        X1    = batch_tok(stemmed_list, 'task1')
        prob1 = self.models['task1'].predict(X1, batch_size=32, verbose=0)
        aspek = (prob1 >= THRESHOLD).astype(int)

        # Task 2-5: prediksi sentimen semua ulasan
        p2 = self.models['task2'].predict(
            batch_tok(stemmed_list, 'task2'), batch_size=32, verbose=0
        ).flatten()
        p3 = self.models['task3'].predict(
            batch_tok(nonstem_list, 'task3'), batch_size=32, verbose=0
        ).flatten()
        p4 = self.models['task4'].predict(
            batch_tok(nonstem_list, 'task4'), batch_size=32, verbose=0
        ).flatten()
        p5 = self.models['task5'].predict(
            batch_tok(stemmed_list, 'task5'), batch_size=32, verbose=0
        ).flatten()

        results = []
        for i in range(len(rows)):
            r = {
                'Room'    : ('Positif' if p2[i] >= THRESHOLD else 'Negatif') if aspek[i][0] else '-',
                'Hotel'   : ('Positif' if p3[i] >= THRESHOLD else 'Negatif') if aspek[i][1] else '-',
                'Location': ('Positif' if p4[i] >= THRESHOLD else 'Negatif') if aspek[i][2] else '-',
                'Service' : ('Positif' if p5[i] >= THRESHOLD else 'Negatif') if aspek[i][3] else '-',
            }
            results.append(r)

        return results


# Singleton instance
_pipeline = None

def get_pipeline(model_dir: str, tokenizer_dir: str) -> ABSAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ABSAPipeline(model_dir, tokenizer_dir)
        _pipeline.load()
    return _pipeline
