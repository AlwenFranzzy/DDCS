Sistem ini mensimulasikan dan mengoptimalkan penggunaan air PDAM dan air daur ulang menggunakan kombinasi MLPRegressor sebagai model prediktif dan Q-Learning (Model-Assisted) sebagai agen pengambil keputusan.
Proyek ini ditujukan untuk mengurangi konsumsi PDAM dan memaksimalkan penggunaan air hasil daur ulang dalam sebuah environment terkontrol.
________________________________________
Fitur Utama
•	Simulasi environment PDAM–Recycle Water.

•	Pengumpulan dataset transisi menggunakan aksi acak.

•	Training model MLP untuk memprediksi kondisi next_recycled.

•	Q-Learning dengan reward shaping berbasis prediksi MLP.

•	Simulasi kebijakan optimal selama 7 hari.
________________________________________
Cara Menjalankan
1. Install dependencies
pip install numpy pandas scikit-learn matplotlib
2. Jalankan program utama
python main.py
3. Output
Program akan menghasilkan:

•	Dataset hasil collect_data

•	Skor model MLP (R²)

•	Q-learning reward curve

•	Simulasi penggunaan air

•	Grafik visualisasi
________________________________________
Workflow Sistem
1.	collect_data()
Menghasilkan dataset transisi dari environment.
2.	train_mlp()
Melatih MLP untuk memprediksi next_recycled.
3.	q_learning_model_assisted()
Menggunakan reward asli + bonus prediksi MLP untuk mempercepat pembelajaran agen.
4.	simulate_policy()
Menguji kebijakan Q-table selama 7 hari dan menghasilkan metrik penggunaan air.
5.	Visualisasi
Grafik ditampilkan melalui matplotlib.
________________________________________
Interpretasi Hasil
•	Learning Curve:
Trend naik → agen semakin optimal.

•	Recycled Stock Trace:
Menampilkan dinamika stok air daur ulang per jam.

•	Tank Visualization:
Memperlihatkan komposisi PDAM vs Recycle pada awal–tengah–akhir simulasi.

•	Recycle Share:
Semakin tinggi → penggunaan air daur ulang makin efisien.
