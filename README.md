# ☁️ Alokasi Server Cloud — Algoritma Hungarian

**Mata Kuliah:** Riset Operasi (SIS3532)  
**Dosen:** Rillya Arundaa, S.Kom, M.Kom  
**Kelompok:** 10 — UNSRAT  
**Topik:** 1 · Kasus 2 — Alokasi Server Cloud Kampus  

---

## 🔗 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 Deskripsi

Aplikasi web interaktif untuk menyelesaikan **Assignment Problem** menggunakan
**Algoritma Hungarian** — mengalokasikan layanan aplikasi kampus ke klaster server
secara optimal untuk meminimalkan total *latency* (ms) atau memaksimalkan kinerja.

**Fitur:**
- ✅ Matriks fleksibel: **8×8 hingga 20×20**
- ✅ Mode **Minimasi** (latency) dan **Maksimasi** (kinerja)
- ✅ Input **acak** (random seed) atau **manual** (data editor)
- ✅ **Transparansi step-by-step** setiap transformasi matriks
- ✅ **Heatmap** matriks + visualisasi bar chart + network map
- ✅ **Verifikasi brute-force** otomatis (N ≤ 9)
- ✅ **Download** CSV dan PNG hasil

---

## 🚀 Cara Deploy ke Streamlit Cloud (Gratis)

### Langkah 1 — Siapkan Repository GitHub

1. Buka [github.com](https://github.com) → **New repository**
2. Nama repo: `hungarian-server-allocation` (atau bebas)
3. Set **Public**
4. Klik **Create repository**

### Langkah 2 — Upload File

Upload 3 file ini ke root repository:
```
hungarian-server-allocation/
├── app.py              ← kode utama Streamlit
├── requirements.txt    ← daftar library
└── README.md           ← file ini
```

**Cara upload:**
- Di halaman repo GitHub → klik **Add file** → **Upload files**
- Drag & drop ketiga file → klik **Commit changes**

### Langkah 3 — Deploy ke Streamlit Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io)
2. Login dengan akun **GitHub** (gratis)
3. Klik **New app**
4. Isi form:
   - **Repository:** `username/hungarian-server-allocation`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Klik **Deploy!**
6. Tunggu ~2 menit → URL online siap 🎉

### Hasil

URL publik format: `https://username-hungarian-server-allocation-app-xxxx.streamlit.app`

URL ini **permanen dan gratis** selama repository GitHub tetap ada.

---

## 📦 Struktur File

```
app.py              — Aplikasi Streamlit lengkap
requirements.txt    — Library: streamlit, numpy, pandas, matplotlib
README.md           — Dokumentasi ini
```

---

## 🧮 Algoritma

```
Input  : Matriks latency n×n (n = 8..20)
Output : Penugasan optimal assignment[i] = j

1. Jika Maksimasi: konversi → max_val − sel
2. Row Reduction   : mat[i] -= min(mat[i])
3. Column Reduction: mat[:,j] -= min(mat[:,j])
4. Iterasi:
   a. Hitung garis penutup minimum (König's Theorem)
   b. Jika garis = n → OPTIMAL, baca penugasan
   c. Cari θ = min sel tidak tertutup
   d. Sel bebas −θ ; perpotongan +θ → ulangi

Kompleksitas: O(n³)
```

---

## 📸 Screenshot

| Tab | Isi |
|-----|-----|
| 📊 Input Matriks | Heatmap matriks + tombol generate/run |
| ✅ Hasil Optimal | Metrics, tabel, bar chart, network map |
| 📋 Langkah-Langkah | Setiap step transformasi + heatmap per langkah |
| 📖 Teori & Panduan | Penjelasan algoritma + cara pakai |
