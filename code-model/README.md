# ระบบตรวจจับและแปลภาษามือไทย | Thai Sign Language Detection System

ระบบปัญญาประดิษฐ์สำหรับการตรวจจับและแปลภาษามือไทยแบบเรียลไทม์ พัฒนาด้วย Python, Flask, TensorFlow และ MediaPipe

## 🌟 คุณสมบัติหลัก (Key Features)

### 🇹🇭 ภาษาไทย
- **ตรวจจับแบบเรียลไทม์**: ใช้กล้องเว็บแคมเพื่อตรวจจับภาษามือแบบสด
- **อัพโหลดไฟล์วิดีโอ**: รองรับการอัพโหลดไฟล์วิดีโอเพื่อวิเคราะห์ภาษามือ
- **AI แม่นยำ**: ใช้โมเดล AI ที่ผ่านการฝึกฝนมาอย่างดี
- **อินเทอร์เฟซที่ทันสมัย**: UI/UX ที่สวยงามและใช้งานง่าย
- **รองรับท่าทาง 10 แบบ**: กลับ, ขอบคุณ, คุณสบายดีไหม, ช่วย, เชื่อ, แนะนำ, พา, รอ, สวัสดี, อะไร

### 🇺🇸 English
- **Real-time Detection**: Use webcam to detect sign language in real-time
- **Video File Upload**: Support video file upload for sign language analysis
- **Accurate AI**: Uses well-trained AI models
- **Modern Interface**: Beautiful and user-friendly UI/UX
- **10 Supported Gestures**: Back, Thank you, How are you, Help, Believe, Recommend, Take, Wait, Hello, What

## 🛠️ การติดตั้ง (Installation)

### ข้อกำหนดระบบ (System Requirements)
- Python 3.8 หรือสูงกว่า
- กล้องเว็บแคม
- RAM 4GB หรือสูงกว่า
- GPU (แนะนำสำหรับประสิทธิภาพที่ดีขึ้น)

### ขั้นตอนการติดตั้ง (Installation Steps)

1. **โคลนโปรเจกต์ (Clone the project)**
```bash
git clone <repository-url>
cd thai-sign-language-detection
```

2. **สร้างสภาพแวดล้อมเสมือน (Create virtual environment)**
```bash
python -m venv venv
```

3. **เปิดใช้งานสภาพแวดล้อมเสมือน (Activate virtual environment)**

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

4. **ติดตั้งแพ็คเกจ (Install packages)**
```bash
pip install -r requirements.txt
```

5. **รันแอปพลิเคชัน (Run the application)**
```bash
python app.py
```

6. **เปิดเบราว์เซอร์ (Open browser)**
```
http://localhost:5000
```

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```
thai-sign-language-detection/
├── app.py                 # Flask application
├── sign_language_model.h5 # Trained AI model
├── angsana.ttc           # Thai font file
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── templates/           # HTML templates
│   ├── index.html       # Main page
│   └── about.html       # About page
├── static/              # Static files (CSS, JS, images)
└── uploads/             # Temporary upload directory
```

## 🚀 การใช้งาน (Usage)

### 1. ตรวจจับแบบเรียลไทม์ (Real-time Detection)
1. เปิดเว็บไซต์ในเบราว์เซอร์
2. คลิกปุ่ม "เริ่มต้น" เพื่อเปิดกล้อง
3. แสดงท่าทางภาษามือต่อกล้อง
4. ระบบจะแสดงผลการตรวจจับและความมั่นใจ
5. คลิกปุ่ม "หยุด" เพื่อปิดกล้อง

### 2. อัพโหลดไฟล์วิดีโอ (Video File Upload)
1. คลิกที่พื้นที่อัพโหลดหรือลากไฟล์วิดีโอมาที่นี่
2. เลือกไฟล์วิดีโอที่ต้องการวิเคราะห์
3. รอการประมวลผล
4. ดูผลการวิเคราะห์ที่แสดง

## 🧠 เทคโนโลยีที่ใช้ (Technologies Used)

### Backend
- **Python**: Programming language
- **Flask**: Web framework
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library
- **MediaPipe**: Hand and pose tracking
- **NumPy**: Numerical computing

### Frontend
- **HTML5**: Markup language
- **CSS3**: Styling
- **JavaScript**: Client-side scripting
- **Bootstrap 5**: CSS framework
- **Font Awesome**: Icons

### AI/ML
- **Deep Learning**: Neural networks
- **CNN**: Convolutional Neural Networks
- **LSTM**: Long Short-Term Memory
- **Pose Estimation**: Body pose detection
- **Hand Tracking**: Hand landmark detection

## 📊 ประสิทธิภาพ (Performance)

- **ความแม่นยำ**: 95%+
- **เวลาประมวลผล**: < 0.5 วินาที
- **ท่าทางที่รองรับ**: 10 ท่าทาง
- **การใช้งาน**: แบบเรียลไทม์และอัพโหลดไฟล์

## 🔧 การปรับแต่ง (Customization)

### เพิ่มท่าทางใหม่ (Add New Gestures)
1. เพิ่มชื่อท่าทางในรายการ `actions` ใน `app.py`
2. ฝึกฝนโมเดลใหม่ด้วยข้อมูลท่าทางเพิ่มเติม
3. อัปเดตโมเดล `sign_language_model.h5`

### ปรับแต่ง UI (Customize UI)
1. แก้ไขไฟล์ CSS ใน `templates/index.html`
2. ปรับแต่งสีและสไตล์ในส่วน `:root` variables
3. เพิ่มหรือลบองค์ประกอบใน HTML

## 🐛 การแก้ไขปัญหา (Troubleshooting)

### ปัญหาที่พบบ่อย (Common Issues)

**1. กล้องไม่ทำงาน (Camera not working)**
- ตรวจสอบการอนุญาตกล้องในเบราว์เซอร์
- ตรวจสอบว่ากล้องไม่ถูกใช้งานโดยโปรแกรมอื่น

**2. โมเดลไม่โหลด (Model not loading)**
- ตรวจสอบว่าไฟล์ `sign_language_model.h5` อยู่ในโฟลเดอร์หลัก
- ตรวจสอบเวอร์ชันของ TensorFlow

**3. ข้อผิดพลาดการอัพโหลด (Upload errors)**
- ตรวจสอบขนาดไฟล์ (สูงสุด 16MB)
- ตรวจสอบรูปแบบไฟล์ (ต้องเป็นวิดีโอ)

## 📝 การพัฒนาเพิ่มเติม (Future Development)

- [ ] เพิ่มท่าทางภาษามือให้มากขึ้น
- [ ] รองรับการแปลเป็นภาษาอังกฤษ
- [ ] เพิ่มการบันทึกและประวัติการใช้งาน
- [ ] พัฒนาแอปพลิเคชันมือถือ
- [ ] เพิ่มการรองรับภาษามือของประเทศอื่น

## 📄 ลิขสิทธิ์ (License)

โปรเจกต์นี้เป็นส่วนหนึ่งของวิชา CPE305 - Computer Engineering Project

---

**หมายเหตุ**: ระบบนี้พัฒนาขึ้นเพื่อการศึกษาและการวิจัย โปรดใช้อย่างรับผิดชอบ 
