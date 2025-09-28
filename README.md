# Face Recognition Service

Layanan API berbasis FastAPI untuk deteksi dan verifikasi wajah menggunakan OpenCV dan DeepFace. Proyek ini dikemas dalam Docker agar mudah dijalankan di berbagai lingkungan.

## Fitur Utama
- Deteksi wajah dari gambar base64.
- Verifikasi apakah dua gambar berisi wajah yang sama.
- Endpoint health check (`/health`) dan informasi layanan (`/info`).
- Registrasi otomatis ke Consul (opsional, aktif bila Consul tersedia).

## Struktur Proyek
- app/main.py: Aplikasi FastAPI beserta endpoint.
- requirements.txt: Dependensi Python 3.
- Dockerfile: Image Docker untuk layanan ini.
- .env: Variabel lingkungan untuk konfigurasi layanan.

## Prasyarat
- Python 3.9 atau lebih baru (untuk menjalankan langsung).
- Gunakan `python3 -m pip` untuk instalasi dependensi Python 3.
- Docker (opsional, jika ingin menjalankan via container).
- Akses jaringan ke Consul (opsional, bila ingin memanfaatkan pendaftaran layanan).

## Konfigurasi Lingkungan
1. Salin dan sesuaikan variabel pada file .env. Nilai standar sudah disertakan.
2. Saat menjalankan di container, pastikan SERVICE_HOST=0.0.0.0 agar aplikasi menerima koneksi dari luar container. Anda bisa menimpanya saat menjalankan docker run.

## Menjalankan Secara Lokal
1. (Opsional) Buat dan aktifkan virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
   Di Windows gunakan .venv\Scripts\activate.
2. Instal dependensi:
    ```bash
    python3 -m pip install --no-cache-dir -r requirements.txt
    ```
3. Jalankan layanan:
    ```bash
    python3 app/main.py
    ```
   Atau dengan hot-reload selama pengembangan:
    ```bash
    python3 -m uvicorn app.main:app --host 0.0.0.0 --port 5001 --reload
    ```
4. Akses API di http://127.0.0.1:5001 (atau host/port sesuai .env).

## Menjalankan dengan Docker
1. Bangun image:
    ```bash
    docker build -t face-recognition-service .
    ```
2. Jalankan container menggunakan file .env yang ada:
    ```bash
    docker run --rm --env-file .env -e SERVICE_HOST=0.0.0.0 -p 5001:5001 --name face-recognition-service face-recognition-service
    ```
   Parameter -e SERVICE_HOST=0.0.0.0 memastikan aplikasi bind ke seluruh interface di dalam container.
3. Buka http://localhost:5001/health untuk memastikan layanan berjalan.
4. Hentikan container dengan Ctrl+C (jika berjalan di foreground) atau:
    ```bash
    docker stop face-recognition-service
    ```

## Menguji Endpoint
- Health Check
    ```bash
    curl http://localhost:5001/health
    ```
- Deteksi Wajah
    ```bash
    curl -X POST http://localhost:5001/detect -H "Content-Type: application/json" -d '{"image": "<BASE64_IMAGE>"}'
    ```
    Ganti <BASE64_IMAGE> dengan string base64 yang hanya berisi data gambar (tanpa prefix data:image/...).
- Verifikasi Wajah
    ```bash
    curl -X POST http://localhost:5001/verify -H "Content-Type: application/json" -d '{"img1": "<BASE64_IMAGE_1>", "img2": "<BASE64_IMAGE_2>"}'
    ```

## Catatan Tambahan
- Endpoint detect dan verify membutuhkan input berupa string base64 gambar. Pastikan gambar memiliki wajah yang jelas agar hasil akurat.
- Jika Consul tersedia, pastikan variabel CONSUL_HOST dan CONSUL_PORT sesuai. Layanan akan otomatis registrasi saat startup dan deregistrasi saat shutdown.
- Dependensi DeepFace cukup besar; pertama kali build image atau instal dependensi mungkin memerlukan waktu lebih lama.
