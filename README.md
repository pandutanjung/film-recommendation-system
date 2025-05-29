# Laporan Proyek Machine Learning - Pandu Persada Tanjung

## Domain Proyek

Di tengah maraknya layanan streaming yang menawarkan ribuan judul film, pengguna kerap mengalami kesulitan dalam memilih tontonan yang sesuai dengan preferensi mereka. Untuk mengatasi tantangan ini, sistem rekomendasi hadir sebagai solusi yang mampu menyaring dan menyuguhkan konten yang relevan secara personal. Khususnya dalam industri hiburan, sistem ini memegang peranan penting dalam meningkatkan keterlibatan dan kenyamanan pengguna. Oleh sebab itu, pengembangan sistem rekomendasi berbasis konten menjadi langkah strategis guna membantu pengguna menemukan film yang sesuai minat mereka, serta mendorong peningkatan kepuasan dan waktu menonton.

## Business Understanding

### Problem Statements
Bagaimana sistem dapat menyarankan film yang relevan dengan preferensi pengguna berdasarkan kesamaan konten?

### Goals
Merancang sistem rekomendasi berbasis konten yang mampu menyarankan film-film dengan kemiripan dari konten dengan film prefensi pengguna.

### Solution statements
- Content-Based Filtering
Sistem ini menganalisis fitur konten seperti genre dan deskripsi film menggunakan metode TF-IDF (Term Frequency–Inverse Document Frequency) untuk merepresentasikan teks secara numerik. Kemudian, Cosine Similarity digunakan untuk menghitung tingkat kemiripan antarfilm. Film dengan skor kemiripan tertinggi akan ditampilkan sebagai rekomendasi kepada pengguna.

## Data Understanding
Sumber Data: [IMDb Movie Dataset](https://github.com/pandutanjung/film-recommendation-system/tree/d8140c80e886f6149452072804b39f4173854389/dataset)

Jumlah Data: 1400

Format:CSV

### Fitur
| Nama | Deskripsi |
| --- | --- |
| name|  Judul film  |
| year | Tahun rilis	 |
| movie_rated | Kategori usia (misal: PG-13, R) |
| run_length | Durasi film |
| genres | Genre film |
| release_date | Tanggal rilis |
| rating | Skor rating penggunan |
| num_raters | Jumlah penilaian |
| num_reviews | Jumlah ulasan |
| review_url | Link ulasan  |

### Exploratory Data Analysis

![Missing & Invalid Values](https://raw.githubusercontent.com/pandutanjung/film-recommendation-system/78a0b4e5c626f0032efb74a900758ada7e8a0639/images/fr-misvalue.png)

Hasil tersebut menunjukkan bahwa tidak ada nilai kosong (missing values) pada seluruh kolom dalam dataset. Setiap kolom memiliki jumlah entri yang lengkap, sehingga data dinyatakan bersih dan siap untuk digunakan dalam analisis atau pemodelan lebih lanjut.

![Duplicate Data](https://raw.githubusercontent.com/pandutanjung/film-recommendation-system/78a0b4e5c626f0032efb74a900758ada7e8a0639/images/fr-duplicate.png)

Perlu ada tindakan untuk menghilangkan duplikasi data

![Descriptive Statistics](https://raw.githubusercontent.com/pandutanjung/film-recommendation-system/78a0b4e5c626f0032efb74a900758ada7e8a0639/images/fr-statdesc.png)

Rating film berkisar antara 3.5 hingga 9.3, dengan nilai rata-rata 7.49, menunjukkan sebagian besar film memiliki penilaian cukup tinggi. Jumlah pemberi rating (num_raters) sangat bervariasi, mulai dari sekitar 19.000 hingga 2,2 juta, dengan rata-rata sekitar 421.000 orang. Sementara itu, jumlah ulasan (num_reviews) berkisar antara 102 hingga 10.279, dengan median 681, menunjukkan adanya perbedaan tingkat popularitas antarfilm.

![Outlier Detection](https://raw.githubusercontent.com/pandutanjung/film-recommendation-system/78a0b4e5c626f0032efb74a900758ada7e8a0639/images/fr-outlier.png)

*   Terlihat bahwa kolom num_raters memiliki banyak outlier (nilai pencilan) yang berada jauh di atas nilai maksimum normal (di luar whisker atas), menunjukkan adanya film-film dengan jumlah rating yang jauh lebih tinggi dibandingkan mayoritas lainnya.

*   Kolom num_reviews, rating, dan year juga mengandung beberapa outlier, meskipun jumlahnya tidak sebanyak num_raters.

*   Distribusi nilai pada rating dan year cukup rapat, menunjukkan konsistensi data pada kolom tersebut.

![Feature Correlation](https://raw.githubusercontent.com/pandutanjung/film-recommendation-system/78a0b4e5c626f0032efb74a900758ada7e8a0639/images/fr-korelasi.png)

* Terdapat korelasi positif kuat antara num_raters dan num_reviews sebesar 0.64, menunjukkan bahwa semakin banyak orang yang memberi rating, cenderung semakin banyak juga yang menulis ulasan.

* Rating memiliki korelasi sedang dengan num_raters (0.56) dan korelasi rendah dengan num_reviews (0.30), yang mengindikasikan bahwa film dengan rating tinggi cenderung lebih populer.

* Korelasi year dengan variabel lain tergolong lemah atau negatif, terutama terhadap rating (-0.23), yang bisa menunjukkan sedikit kecenderungan film lama memiliki rating lebih tinggi dibanding film baru.

## Data Preparation
#### 1. Menghapus Duplikasi Data
Menghilangkan entri yang memiliki nilai duplikat pada kolom name, genres, dan description dengan menggunakan df.drop_duplicates().

#### 2. Validasi Tipe Data Kolom Name
Memastikan bahwa kolom name tidak berisi nilai kosong dengan notnull() dan seluruh nilainya bertipe string melalui astype(str).

#### 3. Menghapus Kolom yang Tidak Diperlukan
Kolom seperti review_url, release_date, dan run_length dihapus menggunakan df.drop() karena dianggap tidak memberikan kontribusi berarti terhadap performa rekomendasi.

#### 4. Encoding pada Kolom Genre
Mengaplikasikan OneHotEncoder dari library scikit-learn untuk mengubah nilai kategorikal dalam kolom genres menjadi format numerik biner, kemudian hasilnya digabungkan kembali ke DataFrame utama.

#### 5. Ekstraksi Fitur Teks Menggunakan TF-IDF
Menggabungkan nilai dari kolom name, genres, dan description menjadi satu string dalam kolom baru bernama features, lalu menggunakan TfidfVectorizer untuk mengubah teks tersebut menjadi vektor numerik.

## Modeling
### Pendekatan
Pada tahap ini, sistem rekomendasi dikembangkan menggunakan pendekatan **content-based filtering**, yang merekomendasikan film berdasarkan kemiripan konten antarfilm. Sistem ini menganalisis fitur seperti genre dan rating dari film yang disukai pengguna, lalu mencari film lain yang memiliki karakteristik serupa. Untuk merepresentasikan fitur secara numerik, digunakan teknik **TF-IDF (Term Frequency-Inverse Document Frequency)** yang menyoroti kata atau genre yang paling unik dalam dataset.

Setelah vektor fitur terbentuk, sistem menghitung **cosine similarity** untuk mengukur tingkat kemiripan antara film yang dipilih dengan film lainnya. Cosine similarity menghasilkan nilai antara 0 hingga 1, di mana nilai yang lebih mendekati 1 menunjukkan kemiripan yang tinggi. Rekomendasi film kemudian dihasilkan berdasarkan film-film dengan nilai kemiripan tertinggi terhadap film input pengguna. Pendekatan ini bersifat individual, karena hanya bergantung pada konten film tanpa mempertimbangkan preferensi pengguna lain.

### Kelebihan dan Kekurangan
#### Kelebihan
1. Personalisasi Tinggi: Rekomendasi disesuaikan langsung dengan preferensi pengguna berdasarkan konten film yang pernah disukai.
2. Tidak Bergantung pada Data Pengguna Lain: Tetap efektif meskipun jumlah pengguna atau interaksi antar pengguna masih sedikit.

#### Kekurangan
1. Kurangnya Keberagaman: Sistem cenderung merekomendasikan film yang terlalu mirip, sehingga pengguna tidak mendapat variasi.
2. Terbatas pada Informasi Fitur – Jika data seperti genre atau deskripsi tidak lengkap, sistem akan kesulitan memberikan rekomendasi yang relevan.

### Hasil Rekomendasi (Inference)

```
Film input: Doctor Strange

Top 5 rekomendasi:
1. Pirates of the Caribbean
2. Thor
3. Star Wars: Episode III - Revenge of the Sith
4. Pirates of the Caribbean: Dead Man's Chest
5. Suicide Squad 

```

## Evaluation
### Metriks Evaluasi
1. **Skor Cosine Similarity:** Menunjukkan tingkat kemiripan antara film yang dimasukkan pengguna dengan film-film yang direkomendasikan.
2. **Precision\@K:** Persentase film yang relevan di antara K film teratas yang direkomendasikan oleh sistem.
3. **Recall\@K:** Persentase film relevan yang berhasil terjaring oleh sistem dari seluruh film relevan yang tersedia.

$$\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}$$

- $\mathbf{A}$ dan $\mathbf{B}$ adalah vektor TF-IDF dari dua film.  
- $\|\mathbf{A}\|$ dan $\|\mathbf{B}\|$ adalah norma (magnitudo) dari masing-masing vektor.  
- $\mathbf{A} \cdot \mathbf{B}$ adalah hasil dot product dari vektor tersebut.

### Hasil Evaluasi Terhadap Metriks
Berdasarkan hasil evaluasi, sistem rekomendasi film berbasis content-based filtering yang dikembangkan mampu menghasilkan rekomendasi dengan tingkat kemiripan konten rata-rata sebesar 0.4279 berdasarkan skor Cosine Similarity. Hal ini menunjukkan bahwa sistem telah cukup berhasil mengenali kesamaan antarfilm berdasarkan fitur seperti genre dan rating.

Meskipun nilai Precision@10 sebesar 0.0482 menunjukkan bahwa hanya sebagian kecil dari rekomendasi yang benar-benar relevan, sistem memiliki Recall@10 sebesar 0.4821, yang mengindikasikan bahwa hampir setengah dari konten relevan berhasil terjaring dalam daftar rekomendasi. Ini menandakan bahwa sistem memiliki cakupan yang cukup baik dalam menemukan film-film dengan karakteristik serupa, meskipun masih perlu peningkatan dalam hal ketepatan hasil.

Secara keseluruhan, sistem ini menunjukkan potensi awal yang menjanjikan dalam membangun rekomendasi berbasis konten. Ke depannya, performa sistem dapat ditingkatkan lebih lanjut dengan memperkaya fitur konten dan mempertimbangkan pendekatan hybrid untuk meningkatkan akurasi dan keberagaman rekomendasi.

### Hasil Evaluasi Terhadap Business Understanding
1. Apakah Sudah Menjawab Problem Statement?
Ya, sistem berhasil memberikan rekomendasi film yang relevan berdasarkan input pengguna dengan mempertimbangkan kemiripan fitur seperti genre dan rating.
2. Apakah Goals Sudah Tercapai?
Ya, sistem rekomendasi telah berhasil dibangun sesuai dengan tujuan yang ditetapkan, meskipun dari segi akurasi masih terdapat ruang untuk perbaikan lebih lanjut.

## Kesimpulan
Proyek ini berhasil membangun sistem rekomendasi film berbasis content-based filtering dengan memanfaatkan teknik TF-IDF dan cosine similarity untuk mengukur kemiripan antarfilm. Sistem mampu memberikan rekomendasi yang relevan berdasarkan input pengguna dengan mempertimbangkan fitur seperti genre dan rating, sehingga problem statement yang diajukan telah terjawab dengan baik.

Tujuan utama proyek juga telah tercapai, yaitu menghasilkan model rekomendasi berbasis konten yang dapat memberikan saran film serupa. Hasil evaluasi menunjukkan bahwa sistem memiliki cakupan rekomendasi yang cukup baik meskipun tingkat akurasinya masih dapat ditingkatkan. 

## Referensi
Fajriansyah M, Adikara PP, Widodo AW. 2021. "Sistem Rekomendasi Film Menggunakan Content Based Filtering". Jurnal Pengembangan Teknologi Informasi dan Ilmu Komputer. 5(6): 2188-2199. 
