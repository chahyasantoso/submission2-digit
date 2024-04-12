# Submission 2: IMDB Movie Review Sentiment Analysis
Nama: Chahya Santoso

Username dicoding: chahya_santoso

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/download?datasetVersionNumber=1) |
| Masalah | Bagaimana mengetahui review negatif terhadap suatu produk (yang dalam hal ini adalah IMDB movie) |
| Solusi machine learning | Mempelajari pola review apakah termasuk review negative/positive. target bisa memprediksi dengan akurasi diatas 80% |
| Metode pengolahan | Metode pengolahan data dengan preprocessing (lowercase, strip linebreak, strip karakter khusus, menghilangkan stopwords, mengubah label 'negative/positive' menjadi 0 dan 1), tokenize dan vectorize text |
| Arsitektur model | Arsitektur model terdiri dari layer textVectorization, layer Embedding dengan dimensi 16, layer GlobalAveragePooling1D, minimal 1 hidden layers dengan dimensi minimal 64 nodes (bisa berubah tergantung tuning), dan output layer dengan dimensi 1. Model menggunakan loss function binary crossentropy dengan metrics BinaryAccuracy dan adam optimizer 
| Metrik evaluasi | Precision, Binary accuracy, loss, false negative, false positive, true negative, true positive |
| Performa model | Model berhasil precision sebesar 87%, binary accuracy sebesar 88% dengan loss sebesar 0.384|
| Opsi deployment | menggunakan server side deployment (menggunakan layanan Cloudeka DekaFlexi) dengan container Docker |
| Web app | [review-sentiment-model](http://103.190.215.213:8501/v1/models/review-sentiment-model/metadata)|
| Monitoring | Prometheus graph run time |
