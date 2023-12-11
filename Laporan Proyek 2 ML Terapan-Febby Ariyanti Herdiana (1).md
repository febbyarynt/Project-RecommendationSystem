# Recommendation System - Febby Ariyanti Herdiana

## Project Overview

![Gambar](https://lp2m.uma.ac.id/wp-content/uploads/2022/04/ManfaatBuku.jpg)

Pada masa ini, membaca adalah modal utama yang harus dikuasai oleh masing-masing  individu. Pribadi yang sering membaca cenderung memiliki pemikiran yang kritis, kreatif, dan inovatif. Setiap individu memiliki kemampuan literasi yang berbeda-beda. Kemampuan tersebut dipengaruhi oleh lingkungan  dimana seseorang tumbuh. Setiap negara memiliki tingkat minat baca yang berbeda-beda. Membaca buku dapat memperluas wawasan sehingga dapat membuka pintu peradaban di masyarkat untuk mencapai kesuksesan. Dengan berbagai manfaat yang didapat dari kebiasaan membaca, maka penting bagi kita untuk meningkatkan literasi dengan membaca buku setiap tahunnya.

Pesatnya perkembangan teknologi digital membuat buku tidak hanya tersedia secara fisik namun juga bisa diakses secara digital dan dapat diakses dimana saja dan kapan saja. Kita dapat mengakses buku melalui aplikasi maupun website yang menyediakan buku secara gratis atau berbayar. Oleh karena itu, sebuah _website_ maupun aplikasi penyedia buku digital membutuhkan suatu sistem yang dapat merekomendasikan buku sesuai dengan preferensi mereka. Tidak hanya untuk meningkatkan kepuasan pengguna terhadap _website_ atau aplikasi, sistem rekomendasi ini juga dapat bermanfaat untuk meingkatkan kebiasaan membaca pengguna yang dapat meningkatkan angka literasi membaca di Indonesia yang pada saat ini masih tergolong rendah. Berdasarkan survei yang dilakukan oleh _Program for International Student Assessment (PISA)_ yang dirilis oleh _Organization for Economic Co-operation and Development (OECD)_ pada tahun 2019, Indonesia menempati peringkat ke 62 dari 70 negara, atau dapat disimpulkan bahwa Indonesia masuk ke dalam 10 negara dengan tingkat literasi yang rendah. Maka dari itu pada project sistem rekomendasi ini penulis akan membuat sistem rekomendasi judul buku menggunakan model Content-Based Filtering dan Collaborative Filtering yang diharapkan dapat memberikan manfaat bagi para pembaca dan mempermudah mereka dalam mencari buku sesuai referensi mereka.

## Business Understanding
---
### Problem Statements

Berdasarkan latar belakang yang telah diuraikan di atas, maka perumusan masalah yang akan diselesaikan pada proyek ini, diantaranya:
* Bagaimana cara melakukan pengolahan data yang baik sehingga dapat digunakan untuk membuat model sistem rekomendasi yang baik?
* Bagaimana cara membuat sistem _machine learning_ yang dapat memberikan sejumlah rekomendasi buku berdasarkan nama penulis buku yang pernah dibaca oleh pengguna? 
* Bagaimana cara membuat sistem _machine learning_ yang dapat memberikan rekomendasi judul buku yang sesuai dengan preferensi pengguna berdasarkan rating yang ada?

### Goals

Berikut adalah tujuan dari pernyataan masalah:
- Melakukan pengolahan data yang baik agar dapat digunakan dalam membangun model sistem rekomendasi buku yang baik.
- Membangun model _machine learning_ dengan sistem rekomendasi buku berdasarkan nama penulis buku yang pernah dibaca oleh pengguna (_user_).
- Membangun model _machine learning_ dengan sistem rekomendasi judul buku yang sesuai dengan preferensi pengguna (_user_) berdasarkan rating yang ada.

### Solution Approach

Solusi yang dapat diterapkan agar goals diatas terpenuhi adalah sebagai berikut:
* Melakukan analisa pada data (_data preparation_) untuk dapat memahami data yang ada seperti memeriksa adanya missing value dan duplikasi data.
* Melakukan pemrosesan pada data seperti normalisasi data rating, agar data dapat dengan mudah di proses oleh model.
* Membangung sistem rekomendasi menggunakan 2 teknik yang umum digunakan yaitu: Content-Based Filtering dan Collaborative Filtering :

1. **_Content Based Filtering_**
   * Ide pendekatan ini adalah merekomendasikan item yang mirip dengan item yang disukai pengguna di masa lalu. Pengembangan model menggunakan pendekatan ini dilakukan untuk menghasilkan rekomendasi buku berdasarkan nama penulis buku yang pernah dibaca oleh pengguna (_user_).

2. **_Collaborative Filtering_**
   * Pendekatan ini berdasarkan pada penilaian komunitas pengguna. Selain itu, ia tidak memerlukan atribut untuk setiap itemnya seperti pada sistem _Content-Based Filtering_. Tujuan pendekatan ini adalah untuk menghasilkan sejumlah rekomendasi judul buku yang sesuai dengan preferensi pengguna (_user_) berdasarkan rating yang telah diberikan sebelumnya.


## Data Understanding
---
Berdasarkan sumber dataset: [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) diperoleh informasi:  

Tabel 1. Informasi Dataset
| Jenis | Keterangan |
| -------- | -------- |
| Sumber Dataset | Book Recommendation Dataset : [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) [3] |
| Owner/Collaborator | Möbius |
| Usability | 10.0 |
| Asal Dataset | [Book-Crossing](https://www.bookcrossing.com/?) |
| Lisensi | [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) |
| Jenis dan Ukuran Berkas | .zip (25 MB) |
| Jumlah File Dataset | 3 File (CSV) |

Berikut ini 3 files dataset, diantaranya 
* Books.csv
* Ratings.csv
* Users.csv

Pada proyek ini penulis hanya akan menggunakan 2 file dataset, yaitu :

**1. Books**
Berikut ini deskripsi variabel yang terdapat pada file **Books.csv**:

* ISBN : Nomor unik buku atau Nomor Buku Standar Internasional
* Book-Title : Judul buku
* Book-Author : Penulis buku
* Year-Of-Publication : Tahun buku diterbitkan
* Publisher : Penerbit buku
* Image-URL-S : URL gambar sampul buku berukuran kecil
* Image-URL-M : URL gambar sampul buku berukuran sedang
* Image-URL-L : URL gambar sampul buku berukuran besar

**2. Ratings**

Berikut ini deskripsi variabel yang terdapat pada file **Ratings.csv**:

* User-ID : ID/kode unik bagi pengguna
* ISBN : Nomor Buku Standar Internasional (_International Standard Book Number_)
* Book-Rating : Rating buku dari user

## **Exploratory Data Analysis (EDA)**

**Univariate Data Analysis**

**1. Books.csv**

Mengecek informasi dari dataframe books menggunakan fungsi `info()`. Berikut adalah output nya.

![](https://github.com/febbyarynt/assets/blob/main/output%20data.png?raw=true)

Berdasarkan output di atas, terlihat bahwa dataframe books memiliki 271360 data entri dan terdapat 8 kolom variabel. Berikut informasi masing-masing variabelnya.

Tabel 2. Informasi Variabel Dataframe Books

| # |	Column	| Non-Null Count	| Dtype |
| -------- | -------- | -------- | -------- |
| 0	| ISBN | 271360 non-null | object |
| 1 |	Book-Title | 271360 non-null | object |
| 2	| Book-Author	| 271359 non-null	| object |
| 3	| Year-Of-Publication	| 271360 non-null	| object |
| 4	| Publisher |	271358 non-null	| object |
| 5 |	Image-URL-S	| 271360 non-null	| object |
| 6	| Image-URL-M	| 271360 non-null	| object |
| 7	| Image-URL-L	| 271357 non-null	| object |

Kemudian melihat distribusi data variabel Year of Publication menggunakan visualisasi data berupa bar.

![](https://github.com/febbyarynt/assets/blob/main/gmbr%20distribusi%20data.png?raw=truez)
Gambar 2. Distribusi Year Of Publication

Dari output di atas dapat terlihat bahwa distribusi data Year of Publication cenderung _right-skewed_.

**2. Ratings**

Mengecek informasi dari dataframe ratings menggunakan fungsi `info()`. Berikut adalah output nya.

![](https://github.com/febbyarynt/assets/blob/main/output%20rating.png?raw=true)

Berdasarkan output di atas, terlihat bahwa dataframe books memiliki 1149780 data entri dan terdapat 3 kolom variabel. Berikut informasi masing-masing variabelnya:

Tabel 3. Informasi Variabel Dataframe Ratings

| # |	Column	| Non-Null Count	| Dtype |
| -------- | -------- | -------- | -------- |
| 0	| User-ID | 1149780 non-null | int64 |
| 1 |	ISBN | 1149780 non-null | object |
| 2	| Book-Rating	| 1149780 non-null	| int64 |

Kemudian melihat distribusi data variabel `Book-Rating` menggunakan visualisasi data berupa bar.

![Distribusi Data Rating](https://github.com/febbyarynt/assets/blob/main/rating.png?raw=true)

Dari visualisasi di atas, diketahui bahwa nilai maksimum rating adalah 10 dan nilai minimumnya adalah 0. Artinya, skala rating berkisar antara 0 hingga 10.

## Data Pre-Processing

---
Di tahap kali ini akan dilakukan _merge_ atau penggabungan data `Books.csv` dan `Ratings.csv` agar pembuatan model lebih efisien. Penggabungan menggunakan fungsi `merge()`. Berikut adalah outputnya.

![](https://github.com/febbyarynt/assets/blob/main/data%20output.png?raw=true)

Berdasarkan output di atas dapat diartikan bahwa setelah digabungkan, kini dataframe kita mempunyai `1032345` baris sampel data dan `10` kolom.

Karena jumlah sample yang terbilang banyak yakni 1032345 baris sampel data, maka pada proyek ini hanya akan diambil 100000 baris sampel data saja.

## Data Preparation

---
Data preparation diperlukan untuk mempersiapkan data agar ketika nanti dilakukan proses pengembangan model diharapkan akurasi model akan semakin baik dan meminimalisir terjadinya bias pada data. Berikut ini merupakan tahapan-tahapan dalam melakukan pra-pemrosesan data:

* **Menyandikan Fitur**
Proses encoding ini kita lakukan karena komputer tidak dapat memproses data bertipe kategori sehingga kita harus mengubah data tersebut menjadi berbentuk bilangan. Proses ini disebut dengan encoding.

* **Melakukan Penanganan Missing Value**
Penanganan yang penulis lakukan pada missing value yaitu dengan melakukan drop data. Tetapi karena dataset yang digunakan cukup bersih, missing value hanya terdapat ketika proses penggabungan dataset.

* **Melakukan Sorting Data Rating berdasarkan ID Pengguna**
Melakukan pengurutan data rating berdasarkan ID Pengguna agar mempermudah dalam melakukan penghapusan data duplikat nantinya.

* **Menghapus Data Duplikat**
Melakukan penghapusan data duplikat agar tidak terjadi bias pada data nantinya.

* **Melakukan Normalisasi Nilai Rating**
Untuk menghasilkan rekomendasi yang sesuai dan akurat maka pada tahap ini diperlukan sebuah normalisasi pada data nilai rating dengan menggunakan formula MinMax pada data rating sebelum memasuki tahap modelling.

* **Melakukan Splitting Dataset**
Untuk melatih model maka penulis perlu melakukan pembagian dataset latih dan juga dataset validasi, untuk dataset latih penulis berikan 80% dari total keseluruhan jumlah data sedangkan dataset validasi sebesar 20% dari keseluruhan data. Hal ini diperlukan untuk pengembangan pada model Collaborative Filtering nantinya.

### **Content Based Filtering**

Berikut tahapan Data Preparation yang dilakukan pada pendekatan ini :

**1. Mengecek dan Menangani Missing Value**

Setelah proses penggabungan menggunakan fungsi `merge`, mari kita cek lagi datanya, apakah ada _missing value_ atau tidak. Pendeteksian _missing value_ dilakukan menggunakan fungsi `isnull()`. Berikut hasil deteksi _missing value_ yang diperoleh:

Tabel 4. Hasil Cek Missing Value
| Variabel |	Jumlah Missing Value	| 
| -------- | -------- | 
| ISBN | 0 |
| book_title | 0 |
| book_author	| 0	|
| year_of_publication	| 0	|
| Publisher |	0	|
| Image_URL_S	| 0	|
| Image_URL_M	| 0	|
| Image_URL_L	| 0	|
| user_id | 4	|
| book_rating	| 4	|

Dari output di atas terlihat bahwa pada variabel "user_id" dan "book_rating" terdapat 4 _missing value_. Untuk mengatasi _missing value_ kita akan menghapusnya menggunakan fungsi `dropna()` dan menampilkan hasilnya. Berikut outputnya.

    99996 rows x 10 columns

Kini dataframe kita memiliki `99996` baris sampel data dan `10` kolom.

**2. Memeriksa dan Menangani Duplikasi Data**

Pertama kita akan membuat variabel `preparation` untuk menampung dataframe hasil tahap sebelumnya yaitu `db_clean`. Kemudian kita akan mengurutkan sampel data berdasarkan `book_title`. Berikut cuplikan outputnya.

Tabel 5. Hasil Pengecekan Data yang Telah Dibersihkan

| | ISBN | book_title | book_author | year_of_publication | Publisher | Image_URL_S | ... |
| -- | ---- | ------ | -------- | -------- | -------- | -------- | --- |
| 36394 | 0307001164 | 101 Dalmatians | Justine Korman | 1996 | Golden Books Publishing Company | http://images.amazon.com/images/P/0307001164... | ... |
| 36394 | 0307001164 | 101 Dalmatians | Justine Korman | 1996 | Golden Books Publishing Company | http://images.amazon.com/images/P/0307001164... | ... |
| 36394 | 0307001164 | 101 Dalmatians | Justine Korman | 1996 | Golden Books Publishing Company | http://images.amazon.com/images/P/0307001164... | ... |
| 36394 | 0307001164 | 101 Dalmatians | Justine Korman | 1996 | Golden Books Publishing Company | http://images.amazon.com/images/P/0307001164... | ... |
| .... | ... | ... | ... | ... | ... | ... | ... |


Dari ouput di atas terlihat banyak data judul buku yang memiliki duplikat, sehingga perlu kita hilangkan duplikatnya menggunakan fungsi `drop_duplicates`, kemudian kita tampilkan hasilnya. 

Berikut output nya :

    2228 rows x 10 columns

Setelah dilakukan data cleaning terhadap data yang duplikat kini pada dataframe terdapat `2228` baris sampel data dan `10` kolom.

**3. Mengubah Dataframe menjadi Sebuah List**

Selanjutnya kita perlu mengubah data series menjadi list menggunakan fungsi `tolist()`. Fungsi `tolist()` ini akan mengubah nilai data dari dataframe menjadi bentuk list dan memasukkan ke dalam variabel yang diassign.

Tahap berikutnya, kita akan membuat `dictionary` untuk menentukan pasangan key-value pada data book_ISBN, title, author, book_year_of_publication, dan book_publisher yang telah kita siapkan sebelumnya.

Dictionary adalah stuktur data yang bentuknya seperti kamus dan berfungsi untuk menyimpan kumpulan data/nilai dengan pendekatan `"key-value"` atau "kunci-nilai". Kata kunci harus unik, sedangkan nilai boleh diisi denga apa saja.

Hal ini kita lakukan agar memudahkan ketika akan mengakses itemnya, yakni menggunakan `key` yang telah ditentukan sebelumnya.

### **Collaborative Filtering**

Berikut tahapan Data Preparation yang dilakukan pada pendekatan ini :

**1. Melakukan encoding pada fitur user_id dan ISBN ke dalam indeks integer**

Proses encoding ini kita lakukan karena komputer tidak dapat memproses data bertipe kategori sehingga kita harus mengubah data tersebut menjadi berbentuk bilangan. Proses ini disebut dengan encoding.

Pada tahap ini kita akan melakukan encoding pada fitur `user_id` dan `ISBN`.

**2. Melakukan pemetaan (_mapping_) pada fitur user_id dan ISBN ke dataframe yang berkaitan**

Tahap pemetaan data adalah proses untuk mengintegrasikan bidang dari banyak kumpulan data ke dalam desain, atau database terpusat. 

Pada proyek ini, pemetaan data diperlukan untuk mentransfer, menggunakan, memproses, dan mengelola data. Tujuan utamanya adalah untuk menggabungkan banyak set data menjadi satu yang unik.

**3. Mengecek jumlah user, jumlah ISBN, dan mengubah nilai rating menjadi float**

* Mengecek data menggunakan fungsi `len()`.
* Mengubah nilai data menggunakan fungsi `astype()`.

Berikut cuplikan outputnya
    
    Jumlah User: 105283, Jumlah ISBN: 340556, Min Rating: 0.0, Max Rating: 10.0

**4. Membagi Data untuk Training dan Validasi**

Proses membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Proses ini sebaiknya dilakukan di awal sebelum proses lainnya [[4]](https://www.oreilly.com/library/view/hands-on-predictive-analytics/9781789138719/), hal ini bertujuan agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 

Pada pendekatan ini kita hanya akan menggunakan sejumlah `10000` sampel data dari seluruh dataframe. 

Selanjutnya, kita bagi data train dan validasi dengan komposisi 80:20. Namun sebelumnya, kita perlu melakukan :
* Pemetaan (mapping) data user dan books menjadi satu value terlebih dahulu.
* Buatlah rating dalam skala 0 sampai 1 agar mudah dalam melakukan proses training. 

## Modeling and Result
---

Modeling pada proyek ini dilakukan menggunakan 2 metode, antara lain : 

### **Content Based Filtering**

* Kelebihan:
  * Semakin banyak informasi yang diberikan pengguna, semakin baik akurasi sistem rekomendasi.
  * Teknik ini baik dipakai ketika skala pengguna (_user_) yang besar.
  * Teknik ini dapat menemukan ketertarikan spesifik dari seorang pengguna (_user_), dan dapat merekomendasikan item yang jarang disukai orang lain.

* Kekurangan:
  * Hanya dapat digunakan untuk fitur yang sesuai, seperti film, dan buku.
  * Tidak mampu menentukan profil dari user baru.
  * Karena _meta feature_ yang digunakan kita yang menentukan sendiri, kualitas dari rekomendasi tergantung kualitas dari _meta feature_ itu sendiri.

**1. Menggunakan TF-IDF Vectorizer**

* Pada pemodelan dengan _Content-Based Filtering_ ini, teknik TF-IDF Vectorizer akan digunakan pada sistem rekomendasi untuk menemukan representasi fitur penting dari setiap nama penulis buku (author).

* TF-IDF atau _Term Frequency-Inverse Document Frequency_ berfungsi untuk mengukur seberapa penting suatu kata terhadap kata-kata lain yang ada dalam dokumen.

* Penerapannya menggunakan fungsi `TfidfVectorizer()`.

**2. Melakukan fit dan transformasi**

* Selanjutnya, lakukan fit dan transformasi ke dalam bentuk matriks menggunakan fungsi `fit_transform` dan menampilkan hasilnya.

Berikut output nya :
   
    (2228, 2089)

Keterangan output :

Berdasarkan output di atas dapat diartikan bahwa pada tfidf_matrix terdapat 2228 ukuran data dan 2089 nama penulis buku (author).

**3. Menghasilkan vektor tf-idf dalam bentuk matriks**

* Pada tahap ini menggunakan fungsi `todense()`.

Berikut outputnya :
   (2228, 2089)

**4. Menghitung Derajat kesamaan menggunakan Cosine Similarity**

Untuk menghitung derajat kesamaan (similarity degree), penulis menggunakan teknik cosine similarity dengan fungsi cosine_similarity dari library sklearn. Berikut dibawah ini adalah rumusnya:
![](https://i2.wp.com/blog.knoldus.com/wp-content/uploads/2019/04/cos_similarity.jpg?w=810&ssl=1)


**5. Melihat matriks kesamaan setiap buku**

Mari kita lihat matriks kesamaan setiap buku dengan menampilkan judul buku dalam 5 sampel kolom (axis = 1) dan 10 sampel baris (axis=0). Berikut cuplikan outputnya : 

| title | OLD MAN AND THE SEA | Angels &amp; Insects : Two Novellas | Eeyore's Little Book of Gloom | Starfire (Bantam Spectra) | ... |
| ---- | -------- | -------- | -------- | -------- | -------- |
| Now You See Her | 0.0 | 0.0 | 0.0 | 0.0 | ... |
| The Subtle Knife (His Dark Materials, Book 2) | 0.0 | 0.0 | 0.0 | 0.0 | ... |
| A Little Honesty: Trials and Triumphs of a Prince of Balona | 0.0 | 0.0 | 0.0 | 0.0 | ... |
| The Red Tent (Bestselling Backlist) | 0.0 | 0.0 | 0.0 | 0.0 | ... |
| .... | ... | ... | ... | ... |

**Keterangan :**

Angka 1.0 mengindikasikan bahwa buku pada kolom X (horizontal) memiliki kesamaan dengan buku pada baris Y (vertikal), dan sebaliknya.

**6. Mendapatkan Rekomendasi**

Pada proyek ini kita akan membuat fungsi `author_recommendation` untuk mendapatkan rekomendasi judul buku berdasarkan nama penulis (author) buku dengan k sebagai jumlah rekomendasi. 

Pada fungsi tersebut akan ada beberapa parameter, antara lain :

* Title : Judul Buku (index kemiripan dataframe).
* Similarity_data : Dataframe mengenai similarity yang telah kita definisikan sebelumnya.
* Items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah `title` dan `author`.
* k : Banyak rekomendasi yang ingin diberikan, disini kita akan memberi nilai `k=5` untuk memberikan 5 rekomendasi pada output sistem. 

Selanjutnya, mari kita terapkan fungsi di atas untuk mendapatkan rekomendasi.

Sebagai contoh, buku yang sudah dibaca adalah "Sudden Prey" yang ditulis oleh John Sandford.

Berikut penerapannya.

    # Mendapatkan rekomendasi judul buku berdasarkan nama penulis (author) dari buku yang berjudul Sudden Prey
    author_recommendations('Sudden Prey')

Berikut output hasil Rekomendasi :

| # |	title	| author	| 
| -------- | -------- | -------- |
| 0	| The Night Crew | John Sandford | 
| 1 |	Chosen Prey	 | John Sandford |
| 2	| The Testament	| John Grisham	| 
| 3	| The Last Juror	| John Grisham	|
| 4	| The Street Lawyer	 |	JOHN GRISHAM	| 

Melalui ouput di atas sistem telah memberikan rekomendasi 5 judul buku berdasarkan kata kunci nama author, yakni "John".


### **Collaborative Filtering_**

* Kelebihan :
  * Tidak memerlukan atribut untuk setiap itemnya.
  * Dapat membuat rekomendasi tanpa harus selalu menggunakan dataset yang lengkap.
  * Unggul dari segi kecepatan dan skalabilitas.
  * Rekomendasi tetap akan berkerja dalam keadaan dimana konten sulit dianalisi sekalipun.

* Kekurangan :
  * Membutuhkan data dari preferensi pengguna, misalnya atribut rating. Jika ada item baru (belum ada rating) maka sistem tidak akan merekomendasikan item tersebut.
 
Berikut tahapan dalam pengembangan model dengan _Collaborative Filtering_.

**1. Proses Training Model**

* Setelah proses data preparation selesai, kita lanjut pada proses training model.
* Pada tahap ini kita membuat sistem menggunakan `class RecommenderNet` dan melakukan proses embedding. Dengan Menginisialisasi model RecommenderNet disini saya melakukan proses embedding terhadap data user dan books. Selanjutnya, melakukan operasi perkalian dot product antara embedding user dan books serta menambahkan bias untuk setiap user dan place. Skor kecocokannya ditetapkan dalam skala [0,1] dengan fungsi aktivasi sigmoid melalui Class RecommenderNet
* Kemudian tahap compile model menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer, dan root mean squared error (RMSE) sebagai metrics evaluation. 
* Latih model menggunakan fungsi `fit()`.

**2. Mendapatkan Rekomendasi**

* Untuk mendapatkan rekomendasi judul buku, pertama kita ambil sampel user secara acak dan definisikan variabel "book_never_read" yang merupakan daftar judul buku yang belum pernah dibaca oleh pengguna.
* Selanjutnya, untuk memperoleh rekomendasi judul, gunakan fungsi `model.predict()` dari library Keras.
* Berikut output hasil rekomendasi dari sistem.

Hasil Rekomendasi Buku dengan Rating Tinggi dari Pengguna

| Top 10 Book Recommendation |
| -------- |
| I Spy Spooky Night: A Book of Picture Riddles (I Spy Books) : Walter Wick | 
| Les Fleurs Du Mal : C. Baudelaire |
| Sunwing (Aladdin Fantasy) : Kenneth Oppel	|
| One Fish Two Fish Red Fish Blue Fish (I Can Read It All by Myself Beginner Books) : DR SEUSS |
| Where the Sidewalk Ends : Poems and Drawings : Shel Silverstein |
| The Power of Myth (Illustrated Edition) : Joseph Campbell |
| Fg on Our Immigrant Ance : J Smith |
| CHILD IS BORN, A : LENNART NILSSON |
| Ender's Shadow : Orson Scott Card |
| Yeats Is Dead! (Vintage Crime/Black Lizard) : Joseph O'Connor |

* Melalui ouput di atas sistem telah memberikan rekomendasi 10 judul buku sesuai preferensi pengguna dan yang memiliki rating tertinggi berdasarkan penilaian yang diberikan oleh pengguna. 


## Evaluation
---
Evaluasi yang akan dilakukan diproyek ini yaitu evaluasi dengan Precision Content untuk Content-Based Filtering dan Root Mean Squared Error (RMSE) untuk Collaborative Filtering.

**1. Content Based Filtering**

Untuk evaluasi dari sistem rekomendasi dengan pendekatan content based filtering kita dapat menggunakan salah satu metric yaitu precision@K. Apa itu precision? Precision adalah perbandingan antara True Positive (TP) dengan banyaknya data yang diprediksi positif. Atau juga bisa ditulis secara matematis sebagai berikut :

precision = TP / (TP + FP)

dimana : TP = True Positive atau positif yang sebenarnya FP = False Positive atau positif yang salah dari hasil prediksi

Namun pada sistem rekomendasi kita tidak akan menggunakan True positive atau False Positive melainkan rating yang diberikan pada buku untuk menentukan buku yang direkomendasikan relevan atau tidak. Dengan rumus sebagai berikut :

precision@K = (# of recommended item that relevan) / (# of recommended item)

Melihat dari hasil rekomendasi yang diberikan, sistem telah memberikan rekomendasi buku berdasarkan kata kunci nama penulis (_author_) dari buku yang pernah dibaca oleh pengguna (_user_). Dalam hal ini terdapat 2 buku yang relevan dengan preferensi pengguna, maka dari hasil rekomendasi yang dihasilkan, didapatkan precision sebagai berikut :

    Precision = 2/5
    Precision = 40%

Hal ini dapat diartikan bahwa sistem rekomendasi yang dibuat memiliki presisi 40% dan sudah bisa memberikan rekomendasi sesuai dengan tujuan dari pengembangan sistem, yakni untuk menghasilkan rekomendasi buku berdasarkan nama penulis buku yang pernah dibaca oleh pengguna (_user_).

**2. Collaborative Filtering**

Untuk evaluasi model yang dibangun dengan pendekatan ini yakni menggunakan metrik evaluasi _Root Mean Squared Error_ (RMSE). RMSE memberi gambaran tentang seberapa banyak kesalahan dalam prediksi yang dibuat oleh sistem. Tujuannya tentu saja untuk mendapatkan eror atau tingkat kesalahan seminimal mungkin. 

---

**Formula :**
![](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT2ZxUnnYzsMg4XgoRb4ePxyxxz8FIfbY7KdSvtmOxqh-aHY_nqE0OR1gbzFj0VIIIxVg&usqp=CAU)


**Cara Kerja :**

RMSE dihitung dengan mengkuadratkan error (prediksi " observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.

Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

---

**Hasil Visualisasi Metrik**

Untuk melihat visualisasi proses training model, mari kita plot metrik evaluasi dengan matplotlib. Berikut hasil visualisasi model.

![Hasil Visualisasi Metrik](https://github.com/febbyarynt/assets/blob/main/rmse.png?raw=true)

Dari hasil plot metrik, didapat kesimpulan :
Proses training model untuk data train cukup smooth dan kemudian menurun secara signifikan hingga nilai error akhir sebesar 0.2837. Sedangkan pada data validasi, nilai error tidak menurun secara signifikan, dan error akhirnya sebesar 0.5234. Walaupun nilai error yang didapat cukup baik untuk sistem rekomendasi, namun model ini masih _underfitting_.

Sebagai kesimpulan dari hasil proyek ini adalah model untuk menampilkan sistem rekomendasi dapat dibangun dengan baik dengan dengan menggunakan dua pendekatan yaitu Content-Based Filtering dan Collaborative Filtering. Pengguna mendapatkan hasil rekomendasi berdasarkan berdasarkan nama penulis buku yang pernah dibaca oleh pengguna dengan model Content-Based Filtering. Dan pengguna mendapatkan hasil rekomendasi 10 buku teratas berdasarkan rating yang telah diberikan sebelumnya. Diharapkan dengan adanya sistem rekomendasi machine learning dapat dikembangkan lebih baik lagi kedepannya sehingga para pengguna dapat lebih antusias lagi dalam membaca buku serta dapat meningkatkan angka literasi di Indonesia.

## Daftar Referensi
---

[1] Romadhon, A. C. (2020). Pentingnya Membaca Dan Menulis Serta Kaitannya Dengan Kemajuan Peradaban Bangsa.

[2] Ilham, Bahrul U. (2022). Harbuknas 2022: Literasi Indonesia Peringkat Ke-62 Dari 70 Negara. Retrieved [16 Oktober 2022] from : [Link](https://bisniskumkm.com/harbuknas-2022-literasi-indonesia-peringkat-ke-62-dari-70-negara/#:~:text=Harbuknas%202022%20%3A%20Literasi%20Indonesia%20Peringkat%20Ke%2D62%20Dari%2070%20negara,-UNESCO%20Menyebut%20indeks)

[3] [Book Recommendation Dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) : Collected by Cai-Nicolas Ziegler in a 4-week crawl (August / September 2004) from the Book-Crossing community with kind permission from Ron Hornbaker, CTO of Humankind Systems. Contains 278,858 users (anonymized but with demographic information) providing 1,149,780 ratings (explicit / implicit) about 271,379 books.

[4] Rhys, Hefin. "Machine Learning with R, the Tidyverse, and MLR". Manning Publications. 2020. Page 286. Tersedia: [O'Reilly Media](https://learning.oreilly.com/library/view/machine-learning-with/9781617296574/).