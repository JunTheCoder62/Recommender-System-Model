# Recommender-System-Model
# Background
Recommender Systems are intelligent systems which are used as an expert in making decisions in real life problems. They have replicated the human experts and positively affected the e-commerce by changing the behavior of customers and sellers. Book Recommender Systems (BRS) help the librarians in the management of library catalog efficiently. It supports the readers in choosing the best book for them. Merchants implement the BRS to manage their inventory and gain more profit. In this paper, we have discussed traditional techniques of recommendation, machine learning techniques and their categories i.e. supervised, unsupervised, semi-supervised and reinforcement learning. Also, Machine Learning (ML) techniques used for the book recommendation and their effect on book recommender systems have been discussed. The work will help the researchers in exploring new dimension for recommendation technology in general and book recommendation in particular.

# Business Understanding

## Problem Statement
- Bagaimana cara membuat model machine learning untuk rekomendasi buku menggunakan metode Content - Based Filtering yang diinginkan oleh user ?
- Bagaimana Evaluasi model yang dibuat dengan menggunakan metode Content - Based Filtering ?

## Goals
- Pembuatan Model Machine Learning dengan menggunakan metode Content - Based Filtering untuk Rekomendasi buku.
- Evaluasi model dengan metode Content - Based Filtering agar model akurat.

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










