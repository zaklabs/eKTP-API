
# eKTP Detection - API

Repositori ini berfungsi sebagai template layanan API untuk memungkinkan pengguna untuk mendeteksi Kartu Tanda Penduduk Elektronik (eKTP) Indonesia. Dengan memanfaatkan teknologi YOLOv8 (You Only Look Once) API ini mampu mendeteksi objek eKTP pada sembarang gambar secara otomatis.



### Fitur Utama:

- Deteksi Objek: Menggunakan model YOLOv8 untuk mendeteksi objek eKTP pada sembarang gambar.
- Respon Cepat: Dikembangkan dengan FastAPI, memberikan performa tinggi dan respon cepat untuk setiap permintaan.
- Integrasi Mudah: Dapat dengan mudah diintegrasikan dengan aplikasi lain melalui antarmuka RESTful API.
- Docker: sebuah platform untuk membangun, mengirimkan, dan menjalankan aplikasi terdistribusi dengan mudah

# Mari mulai
Anda dapat memulai dengan 2 cara, menggunakan Docker ataupun secara Local di mesin anda.

## Docker
Mulai aplikasi dengan perintah berikut:
``` 
docker-compose up -d
```

## Local
Untuk memulai secara lokal, ikuti langkah-langkah berikut:
1. Buatkan enviroment python:
```
python -m venv <nama_virtual_enviroment> 
```
2. Aktifkan enviroment yang sudah di buat:
```
source <path_virtual_enviroment>/bin/activate
```
3. Install kebutuhan packages:
```
pip install -r requirements.txt
```
4. Start aplikasi:
```
python main.py
```


# FastAPI Swagger Docs:
Silahkan akses link berikut:
http://127.0.0.1:8004/docs#/. atau ganti url tersebut sesuai dengan ip host mesin anda.
![App Screenshot](https://github.com/zaklabs/eKTP-API/blob/main/assets/api-docs.jpg?raw=true)



#### Get info the machine
```http
  GET /
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `none` | - | - |

#### Get status machine

```http
  GET /api/v1/healthcheck
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `none` | - | - |

#### Get image detection

```http
  POST /api/v1/detection
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `image`      | `file` | **Required**. file to process |

## Demo

Kode berikut menunjukkan cara melakukan deteksi objek dan menerima hasilnya dalam format gambar.

![App Screenshot](https://github.com/zaklabs/eKTP-API/blob/main/assets/detection.jpg?raw=true)


# Hubungi kami
[Telegram Group](https://t.me/+KXLY8hK8VKc5ODM1)
