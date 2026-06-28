# ใช้ Python 3.9 (หรือเวอร์ชันที่ตรงกับที่คุณใช้พัฒนา)
FROM python:3.9-slim

# กำหนดสิทธิ์สำหรับ Hugging Face Spaces (จำเป็นต้องรันด้วย user 'user')
RUN useradd -m -u 1000 user

# ติดตั้ง System dependencies สำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# เปลี่ยน user และตั้งค่า working directory
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# คัดลอก requirements และติดตั้ง
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์ทั้งหมดในโปรเจกต์เข้ามา
COPY --chown=user . .

# เปิดพอร์ต 7860 (Hugging Face Spaces บังคับใช้พอร์ตนี้)
EXPOSE 7860

# รัน Gunicorn เซิร์ฟเวอร์
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]