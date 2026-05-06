# HotelAnalyzer AI — Panduan Instalasi Lengkap

## Struktur Project
```
hotel_analyzer/
├── app.py                 ← File utama Flask (jalankan ini)
├── requirements.txt       ← Library yang harus diinstall
├── README_INSTALASI.md    ← Panduan ini
├── models/                ← Simpan file model .h5 di sini
│   ├── task1_aspek.h5
│   ├── task2_room.h5
│   ├── task3_hotel.h5
│   ├── task4_location.h5
│   └── task5_service.h5
├── tokenizers/            ← Simpan tokenizer .pkl di sini
│   ├── tokenizer_stem.pkl
│   └── tokenizer_nonstem.pkl
├── templates/             ← HTML halaman web (jangan diubah)
├── static/                ← CSS, JS, gambar (jangan diubah)
└── utils/
    ├── preprocessing.py   ← Pipeline preprocessing teks
    └── pipeline.py        ← Pipeline prediksi ABSA

```

---

## STEP 1 — Download File dari Google Drive

### Model .h5 (simpan di folder models/)
Download 5 file model dari Google Drive dan RENAME seperti ini:

| File di Google Drive | Rename menjadi |
|---------------------|----------------|
| Task1_Aspek/model_final.h5 | task1_aspek.h5 |
| Task2_Room/model_final.h5 | task2_room.h5 |
| Task3_Hotel/model_final.h5 | task3_hotel.h5 |
| Task4_Location/model_final.h5 | task4_location.h5 |
| Task5_Service/model_final.h5 | task5_service.h5 |

### Tokenizer .pkl (simpan di folder tokenizers/)
Download 2 file tokenizer:

| File di Google Drive | Nama di folder tokenizers/ |
|---------------------|---------------------------|
| tokenizer_stem.pkl | tokenizer_stem.pkl |
| tokenizer_nonstem.pkl | tokenizer_nonstem.pkl |

---

## STEP 2 — Install Python dan Library

### Pastikan Python sudah terinstall
Buka Command Prompt / Terminal, ketik:
```
python --version
```
Harus muncul Python 3.9 atau lebih baru.

### Buat Virtual Environment (sangat disarankan!)
Di terminal, masuk ke folder project:
```
cd hotel_analyzer
python -m venv venv
```

Aktivasi virtual environment:
- Windows: `venv\Scripts\activate`
- Mac/Linux: `source venv/bin/activate`

### Install semua library
```
pip install -r requirements.txt
```

Library yang diinstall:
- `flask` — web framework
- `tensorflow` — untuk load model deep learning
- `pandas`, `numpy` — olah data
- `PySastrawi` — stemming Bahasa Indonesia
- `indoNLP` — text cleaning bahasa Indonesia
- `langdetect` — deteksi bahasa
- `deep-translator` — translate Inggris → Indonesia
- `openpyxl`, `xlsxwriter` — baca/tulis Excel

---

## STEP 3 — Jalankan Website

Di terminal (pastikan virtual environment aktif):
```
python app.py
```

Atau dengan Flask CLI:
```
flask run --debug
```

Buka browser dan akses:
```
http://localhost:5000
```

---

## PIPELINE ABSA — Penjelasan Teknis

### Alur setiap ulasan:
```
Input Ulasan
    ↓
1. Deteksi bahasa (langdetect)
   - Indonesia → lanjut
   - Inggris → translate ke Indonesia (deep_translator)
   - Bahasa lain → tolak dengan pesan error

2. Cek panjang minimal (< 3 kata → skip)

3. Preprocessing:
   a. text_cleaning (hapus URL, HTML, angka, tanda baca)
   b. case_folding (semua huruf kecil)
   c. normalize_text:
      - frasa domain hotel (check in → checkin, dll)
      - elongasi (baguuus → bagus)
      - custom_norm_dict DULU (nggak→tidak, enggak→tidak)
      - kamus alay GitHub
   d. stopword_removal (Sastrawi - kata wajib pertahankan)
   e. stemming (PySastrawi) → review_stemmed
   f. review_non_stemmed = setelah stopword (tanpa stem)

4. Task 1 — Deteksi Aspek:
   Model: BiLSTM + FastText + STEM
   Input: review_stemmed + tokenizer_stem
   Output: [Room=1/0, Hotel=1/0, Location=1/0, Service=1/0]

5. Task 2 — Sentimen Room (jika Room=1):
   Model: CNN + FastText + STEM
   Input: review_stemmed + tokenizer_stem

6. Task 3 — Sentimen Hotel (jika Hotel=1):
   Model: BiLSTM + Word2Vec + NON-STEM
   Input: review_non_stemmed + tokenizer_nonstem

7. Task 4 — Sentimen Location (jika Location=1):
   Model: CNN + Word2Vec + NON-STEM
   Input: review_non_stemmed + tokenizer_nonstem

8. Task 5 — Sentimen Service (jika Service=1):
   Model: CNN + Word2Vec + STEM
   Input: review_stemmed + tokenizer_stem

Output: Positif / Negatif / - (tidak dibahas)
```

### PENTING: Konsistensi Tokenizer
| Task | Model | Embedding | Teks | Tokenizer |
|------|-------|-----------|------|-----------|
| Task 1 | BiLSTM | FastText | STEM | tokenizer_stem |
| Task 2 | CNN | FastText | STEM | tokenizer_stem |
| Task 3 | BiLSTM | Word2Vec | NON-STEM | tokenizer_nonstem |
| Task 4 | CNN | Word2Vec | NON-STEM | tokenizer_nonstem |
| Task 5 | CNN | Word2Vec | STEM | tokenizer_stem |

---

## TROUBLESHOOTING

### Error: Model tidak ditemukan
→ Pastikan 5 file .h5 ada di folder `models/` dengan nama yang benar

### Error: Tokenizer tidak ditemukan
→ Pastikan 2 file .pkl ada di folder `tokenizers/`

### Error: Library tidak terinstall
→ Jalankan `pip install -r requirements.txt`

### Website lambat pertama kali
→ Normal! Model TensorFlow perlu warm-up untuk request pertama

### Error translate
→ Butuh koneksi internet untuk translate

