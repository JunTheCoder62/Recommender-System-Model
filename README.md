# Background
Sistem Rekomendasi merupakan sistem cerdas yang digunakan sebagai ahli dalam membuat keputusan dalam masalah kehidupan nyata. Mereka telah mereplikasi pakar manusia dan berdampak positif pada e-commerce dengan mengubah perilaku pelanggan dan penjual. Sistem Rekomendasi Buku (BRS) membantu pustakawan dalam mengelola katalog perpustakaan secara efisien. Ini mendukung pembaca dalam memilih buku terbaik untuk mereka. Pedagang menerapkan BRS untuk mengelola inventaris mereka dan mendapatkan lebih banyak keuntungan. Dalam makalah ini, kami telah membahas teknik rekomendasi tradisional, teknik pembelajaran mesin dan kategorinya yaitu pembelajaran terbimbing, tidak terbimbing, semi-terbimbing dan penguatan. Selain itu, teknik Pembelajaran Mesin (ML) yang digunakan untuk rekomendasi buku dan pengaruhnya terhadap sistem rekomendasi buku telah dibahas. Pekerjaan ini akan membantu para peneliti dalam mengeksplorasi dimensi baru untuk teknologi rekomendasi secara umum dan rekomendasi buku secara khusus.

# Business Understanding
## Problem Statement
- Bagaimana cara membuat model machine learning untuk rekomendasi buku menggunakan metode Content - Based Filtering yang diinginkan oleh user ?
- Bagaimana Evaluasi model yang dibuat dengan menggunakan metode Content - Based Filtering ?

## Goals
- Meningkatkan kepuasan pengguna dengan menyediakan rekomendasi buku yang relevan dan sesuai dengan preferensi mereka, sehingga mendorong peningkatan waktu yang dihabiskan di platform.

- Meningkatkan loyalitas pelanggan dan memperkuat retensi pengguna melalui rekomendasi yang lebih personal dan tepat sasaran, yang diharapkan dapat meningkatkan konversi pembelian buku.


## Solution Statement
- Mengembangkan model rekomendasi buku menggunakan metode Content-Based Filtering untuk menyarankan buku berdasarkan kesamaan atribut (genre, tag, deskripsi) dengan buku yang telah disukai atau dibaca oleh pengguna. Model ini akan mengidentifikasi buku-buku serupa dengan preferensi genres pengguna guna memberikan rekomendasi yang lebih relevan.

- Melakukan evaluasi kinerja model menggunakan metrik evaluasi seperti Precision dan Recall, untuk memastikan rekomendasi yang dihasilkan akurat dan relevan dengan preferensi pengguna, sehingga dapat meningkatkan kepuasan dan loyalitas pengguna. Mengoptimalkan hasil model dengan eksperimen berulang pada parameter dan fitur, guna mencapai akurasi yang lebih tinggi, yang pada akhirnya akan meningkatkan pengalaman pengguna dan mendorong lebih banyak konversi pembelian buku di platform.

# Data Understanding
Source data yang didapat merupakan data yang berasal dari Githun dengan Source Link: [Github](https://github.com/shaido987/novel-dataset/tree/master). Dengan jumlah data sebanyak 21,831. terdiri dari :


| **Category**                   | **Column**                                 | **Description**                                                     |
|--------------------------------|--------------------------------------------|---------------------------------------------------------------------|
| **General Information**        | Novel ID                                   | Identifikasi unik untuk setiap novel                                |
|                                | Name                                       | Judul novel                                                         |
|                                | Associated Names                           | Nama atau judul lain untuk novel                                    |
|                                | Original Language                          | Bahasa asli di mana novel ditulis                                   |
|                                | Author / Authors                           | Penulis novel                                                       |
|                                | Genres                                     | Klasifikasi genre novel                                             |
|                                | Tags                                       | Tag yang terkait dengan tema novel                                  |
| **Publishing Information**     | Start Year                                 | Tahun mulai penerbitan novel                                        |
|                                | Licensed                                   | Apakah novel tersebut berlisensi resmi                              |
|                                | Original Publisher                         | Penerbit dari versi asli                                            |
|                                | English Publisher                          | Penerbit dari versi terjemahan bahasa Inggris                       |
| **Chapter Information**        | Number of Chapters (original language)     | Jumlah total bab dalam bahasa asli                                  |
|                                | Completed (original language)              | Apakah versi asli sudah selesai                                     |
|                                | Number of Chapters (translation)           | Jumlah total bab dalam versi terjemahan                             |
|                                | Completed (translation)                    | Apakah terjemahan sudah selesai                                     |
| **Release Information (translation)** | Release Frequency                   | Frekuensi rilis bab terjemahan                                      |
|                                | Activity Weekly Rank                       | Peringkat mingguan berdasarkan aktivitas                            |
|                                | Activity Monthly Rank                      | Peringkat bulanan berdasarkan aktivitas                             |
|                                | Activity All-time Rank                     | Peringkat sepanjang waktu berdasarkan aktivitas                     |
| **Community Information (translation)** | On Number of Reading Lists     | Jumlah daftar bacaan yang mencakup novel ini                        |
|                                | Reading List Monthly Rank                  | Peringkat bulanan dalam daftar bacaan                               |
|                                | Reading List All-time Rank                 | Peringkat sepanjang waktu dalam daftar bacaan                       |
| **Ratings**                    | Rating                                     | Rata-rata penilaian novel                                           |
|                                | Rating Votes                               | Jumlah suara yang berkontribusi pada penilaian                      |
| **Related Series Information** | Related Series IDs                         | ID dari seri yang terkait dengan novel ini                          |
|                                | Recommended Series IDs                     | ID dari seri yang direkomendasikan berdasarkan novel ini            |
|                                | Recommendation List IDs                    | ID dari daftar rekomendasi yang mencakup novel ini                  |

# EDA
## Univariate Analysis
![Scrdsds](https://github.com/user-attachments/assets/f359d446-a489-495d-aeb6-cdebea521e5a)
Gambar .0 Fitur genres buku
Tabel ini memberikan gambaran tentang kombinasi genre buku yang ada dalam dataset, jumlah data untuk setiap kombinasi, dan persentase dari keseluruhan dataset. fitur genres ini memiliki 9490 baris dan 3 kolom.

# Data Prepartion
## Handling Data
Beberapa kolom mungkin tidak diperlukan dalam analisis atau tujuan spesifik yang ingin dicapai. Misalnya, kolom-kolom yang berkaitan dengan data penerbitan, peringkat aktivitas, atau informasi terkait lainnya mungkin tidak relevan jika fokus analisis adalah pada aspek lain dari dataset. maka beberapa dataset didrop agar meninggkatkan performa dan tidak overfitting pada model.
```python
df = df.drop(['start_year', 'original_language', 'original_publisher',
              'english_publisher', 'complete_original', 'chapters_original_current',
              'complete_translated', 'activity_week_rank', 'activity_month_rank', 'activity_all_time_rank',
              'on_reading_lists', 'reading_list_month_rank', 'recommendation_list_ids', 'chapter_latest_translated',
              'assoc_names', 'reading_list_all_time_rank', 'licensed', 'related_series_ids', 'recommended_series_ids',
              'release_freq'], axis=1)
```
**Sehingga Didapat:** 

| Column        | Non-Null Count | Data Type | Description                     |
|---------------|----------------|-----------|---------------------------------|
| `id`          | 21831          | int64     | ID unik untuk setiap novel      |
| `name`        | 21831          | object    | Nama atau judul dari novel      |
| `authors`     | 21831          | object    | Penulis dari novel              |
| `genres`      | 21831          | object    | Genre novel                     |
| `tags`        | 21831          | object    | Tag yang terkait dengan tema    |
| `rating`      | 21831          | float64   | Rata-rata penilaian novel       |
| `rating_votes`| 21831          | int64     | Jumlah suara untuk penilaian    |

Diatas merupakan data yang akan digunakan sebagai fitur dalam model, dengan metode Content-Based Filtering.


## Handling Missing Value
Pada dataset yang dipilih terdapat `0` missing value sehingga data dapat langsung digunakan untuk model machine learning yang akan digunakan.
| Column         | Missing Values |
|----------------|----------------|
| `id`           | 0              |
| `name`         | 0              |
| `authors`      | 0              |
| `genres`       | 0              |
| `tags`         | 0              |
| `rating`       | 0              |
| `rating_votes` | 0              |


## Vektrorisasi Menggunakan TF-IDF
Dalam sistem rekomendasi buku ini, kolom `genres` digunakan untuk menemukan kemiripan antar buku, yang memungkinkan kita menghitung nilai kesamaan antara satu buku dengan buku lainnya. Untuk mendapatkan nilai kemiripan tersebut, kolom `genres` diubah menjadi vektor menggunakan algoritma TF-IDF.

Dalam model ini, fungsi `TfidfVectorizer()` dari library `Scikit-learn` digunakan untuk mengonversi kolom "genre" pada dataset menjadi vektor, dengan hasil vektorisasi yang ditampilkan pada gambar berikut.

<img width="1101" alt="Screenshot 2024-11-12 165938" src="https://github.com/user-attachments/assets/b1b3f9bd-5b80-47a0-8ea4-8f6e3ec2619f">

Gambar .1 Vektorisasi Menggunakan TF-IDF

# Modelling
## Cosine Similarity
Setelah data dikonversi menjadi bentuk vektor, selanjutnya ukur tingkat kesamaan antara dua vektor dan menentukan apakah kedua vektor tersebut menunjuk ke arah yang sama. Semakin kecil sudut cosinus, semakin besar nilai cosine similarity.

$\cos \theta = \dfrac{A.B}{||A||||B||}$

<img width="1142" alt="Screenshot 2024-11-12 191354" src="https://github.com/user-attachments/assets/7df590ac-1c7b-47c2-971a-d02cf2c9cb96">

Gambar .2 Cosine Similarity

## Recommendations Function
Proses Modelling untuk membuat sebuah model Book Recommendations merupakan fungsi untuk mendapat _top-N Recommendations_. Setelah vektroisasi dari **Cosine Similarity** kita dapat menggunakan fitur `genres`. Hasil Recommendations dapat menerima input berupa judul buku `name` dan `genres`.

<img width="1151" alt="Screenshot 2024-11-12 192343" src="https://github.com/user-attachments/assets/28aaa7bd-15d9-459f-be0c-c556a7b7383c">

Gambar .3 top 5 Recommendations Book Function

# Evaluasi
Dalam project machine learning kali ini pembuatan model dengan menggunakan metode Content-Based Filtering dengan Ventorisasi **TF-IDF** dan **Cosine Similarity** untuk dapat menemukan derajat kemiripan antar `genres` buku dan membuat function untuk menampilkan beberapa rekomendasi buku dengan input judul buku. Pengukuran hasi yang diberikan dengan Precision untuk mengukur keakuratan model dapat diberikan dengan **formula metrik** rumus presisi :

$$
\text{Precision} = \dfrac{\text{of our recommendations that are relevant}}{\text{of the items we recommended}}
$$

Dengan hasil prediksi buku yang diberikan dapat dilihat bahwa model menampilkan top 5 buku yang relevan dengan judul buku yang di input. Maka dapat disimpulkan bahwa model memeiliki _Precision_ sampai 100%.

# Conclusion
Dalam pembuatan projek kali ini dapat disimpulkan bahwa:
- Pembuatan model untuk Book Recommendation dengan Content-Based Filtering dengan menggunakan vektorisasi TfidfVectorizer dan Cosine Similarity untuk menentukan relevansi antara Buku untuk user.
- Evaluasi model menggunakan metode Content-Based Filtering dilakukan untuk memastikan akurasi dan efektivitas model dalam memberikan rekomendasi buku top 5 yang relevan dengan input yang diberikan dengan _Precision_ 100%

# Reference
_Anwar, K., Siddiqui, J., & Saquib Sohail, S. (2019, February). Machine learning techniques for book recommendation: an overview. In Proceedings of International Conference on Sustainable Computing in Science, Technology and Management (SUSCOM), Amity University Rajasthan, Jaipur-India._




