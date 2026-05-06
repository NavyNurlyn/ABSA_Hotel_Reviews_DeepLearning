"""
utils/pipeline.py — dengan confidence score + fix TF retracing
"""

import os
import pickle
import time
import numpy as np
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
except ImportError:
    from keras.src.legacy.preprocessing.sequence import pad_sequences

MAX_LENGTH  = 100
THRESHOLD   = 0.5
ASPEK_NAMES = ['Room', 'Hotel', 'Location', 'Service']

TASK_CONFIG = {
    'task1': 'stem',
    'task2': 'stem',
    'task3': 'nonstem',
    'task4': 'nonstem',
    'task5': 'stem',
}


class ABSAPipeline:

    def __init__(self, model_dir: str, tokenizer_dir: str):
        self.model_dir     = Path(model_dir)
        self.tokenizer_dir = Path(tokenizer_dir)
        self.models        = {}
        self.tokenizers    = {}
        self._loaded       = False
        # Cache input shapes untuk fix retracing
        self._input_specs  = {}

    def load(self):
        print("[Pipeline] Loading tokenizer dan model...")
        t0 = time.time()

        for name in ['tokenizer_stem', 'tokenizer_nonstem']:
            path = self.tokenizer_dir / f'{name}.pkl'
            if not path.exists():
                raise FileNotFoundError(f"Tokenizer tidak ditemukan: {path}")
            with open(path, 'rb') as f:
                self.tokenizers[name] = pickle.load(f)
            vocab = len(self.tokenizers[name].word_index) + 1
            print(f"  OK {name}: vocab={vocab:,}")

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
                    f"Pastikan {fname} ada di folder models/"
                )
            print(f"  Loading {fname}...")
            self.models[task] = tf.keras.models.load_model(
                str(path), compile=False
            )
            print(f"  OK {task}: {fname}")

        # Warmup: jalankan prediksi dummy sekali
        # agar tidak terjadi retracing saat request pertama
        print("  Warmup prediksi...")
        dummy = np.zeros((1, MAX_LENGTH), dtype=np.int32)
        for task in self.models:
            _ = self.models[task].predict(dummy, verbose=0)
        print("  Warmup selesai")

        self._loaded = True
        print(f"[Pipeline] Semua siap dalam {time.time()-t0:.1f}s")

    def _tokenize(self, text: str, task: str) -> np.ndarray:
        text_type = TASK_CONFIG[task]
        tokenizer = self.tokenizers[f'tokenizer_{text_type}']
        seq = tokenizer.texts_to_sequences([text])
        return pad_sequences(
            seq, maxlen=MAX_LENGTH,
            padding='post', truncating='post'
        )

    def predict_single(self, nonstem: str, stemmed: str) -> dict:
        """Prediksi tanpa confidence (untuk kompatibilitas)."""
        hasil, _ = self.predict_single_with_confidence(nonstem, stemmed)
        return hasil

    def predict_single_with_confidence(
        self, nonstem: str, stemmed: str
    ) -> tuple:
        """
        Prediksi dengan confidence score.
        Return: (hasil_dict, confidence_dict)

        confidence_dict contoh:
        {
          'Room'    : 0.92,   # prob aspek dibahas
          'Room_sent': 0.88,  # prob positif (jika dibahas)
          'Hotel'   : 0.05,   # prob aspek dibahas (rendah = tidak dibahas)
          ...
        }
        """
        if not self._loaded:
            raise RuntimeError("Pipeline belum di-load!")

        hasil      = {a: '-' for a in ASPEK_NAMES}
        confidence = {}

        # Task 1
        X1    = self._tokenize(stemmed, 'task1')
        prob1 = self.models['task1'].predict(X1, verbose=0)[0]
        aspek = (prob1 >= THRESHOLD).astype(int)

        # Simpan confidence aspek detection
        for i, nama in enumerate(ASPEK_NAMES):
            confidence[f'{nama}_aspek'] = round(float(prob1[i]), 4)

        # Task 2-5
        task_map = {
            0: ('task2', stemmed),
            1: ('task3', nonstem),
            2: ('task4', nonstem),
            3: ('task5', stemmed),
        }
        for idx, (task_key, teks) in task_map.items():
            nama = ASPEK_NAMES[idx]
            if aspek[idx] == 1:
                X    = self._tokenize(teks, task_key)
                prob = float(self.models[task_key].predict(X, verbose=0)[0][0])
                hasil[nama]                    = 'Positif' if prob >= THRESHOLD else 'Negatif'
                confidence[f'{nama}_sentimen'] = round(prob, 4)
            else:
                confidence[f'{nama}_sentimen'] = None

        return hasil, confidence

    def predict_batch(self, rows: list) -> list:
        if not self._loaded:
            raise RuntimeError("Pipeline belum di-load!")
        if not rows:
            return []

        stemmed_list = [r['stemmed']  for r in rows]
        nonstem_list = [r['nonstem']  for r in rows]

        def batch_tok(texts, task):
            tok  = self.tokenizers[f'tokenizer_{TASK_CONFIG[task]}']
            seqs = tok.texts_to_sequences(texts)
            return pad_sequences(seqs, maxlen=MAX_LENGTH,
                                 padding='post', truncating='post')

        X1    = batch_tok(stemmed_list, 'task1')
        prob1 = self.models['task1'].predict(X1, batch_size=32, verbose=0)
        aspek = (prob1 >= THRESHOLD).astype(int)

        p2 = self.models['task2'].predict(
            batch_tok(stemmed_list, 'task2'), batch_size=32, verbose=0).flatten()
        p3 = self.models['task3'].predict(
            batch_tok(nonstem_list, 'task3'), batch_size=32, verbose=0).flatten()
        p4 = self.models['task4'].predict(
            batch_tok(nonstem_list, 'task4'), batch_size=32, verbose=0).flatten()
        p5 = self.models['task5'].predict(
            batch_tok(stemmed_list, 'task5'), batch_size=32, verbose=0).flatten()

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


_pipeline = None

def get_pipeline(model_dir: str, tokenizer_dir: str) -> ABSAPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ABSAPipeline(model_dir, tokenizer_dir)
        _pipeline.load()
    return _pipeline
