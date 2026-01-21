# ThaiSignLaTranslator

## โครงการวิชา Machine Learning

**ThaiSignLaTranslator** เป็นโครงการที่พัฒนาระบบแปลภาษามือไทย (Thai Sign Language) เป็นข้อความภาษาไทยโดยใช้เทคนิค Machine Learning และ Deep Learning  โดยอ้างอิงจากชุดข้อมูลวิดีโอและภาพของสัญญาณมือไทยที่จัดทำโดย Arucha Khematharonon

### วัตถุประสงค์
- สร้างโมเดลที่สามารถรับอินพุตเป็นวิดีโอหรือภาพของสัญญาณมือไทยและแปลงเป็นข้อความภาษาไทยได้
- ฝึกฝนและประเมินเทคนิคการเรียนรู้เชิงลึกสำหรับงานจำแนกท่ามือ
- นำผลลัพธ์ไปใช้เป็นส่วนหนึ่งของระบบช่วยสื่อสารสำหรับผู้บกพร่องทางการได้ยิน

### ฟีเจอร์หลัก
- **การประมวลผลวิดีโอ**: รองรับการสตรีมจากกล้องเว็บแคม
- **การตรวจจับมือ**: ใช้ MediaPipe Hand Tracking เพื่อตรวจจับตำแหน่งของมือในแต่ละเฟรม
- **โมเดลจำแนกท่า**: โมเดลใช้สถาปัตยกรรม LSTM สำหรับการจำแนกท่าทาง
- **การแปลเป็นข้อความ**: ผลลัพธ์จากโมเดลจะถูกแปลงเป็นข้อความภาษาไทย
- **อินเทอร์เฟซเว็บ**: หน้าเว็บที่สร้างด้วย HTML, CSS และ JavaScript เพื่อให้ผู้ใช้ทดลองอัปโหลดและดูผลลัพธ์แบบเรียลไทม์

### โครงสร้างโฟลเดอร์
```
ThaiSignLaTranslator/
│   README.md               # คู่มือการใช้งาน (ไฟล์นี้)
│   requirements.txt        # รายการไลบรารี Python ที่ต้องติดตั้ง
│   app.py                  # Flask แอปหลัก
│   model/                  # โมเดลที่ฝึกแล้ว (.h5, .pth)
│   static/                 # ไฟล์สไตล์ CSS, JS, รูปภาพ UI
└── templates/             # HTML templates สำหรับเว็บ
```

### วิธีติดตั้งและใช้งาน
1. **คล cloning โครงการ**
   ```bash
   git clone https://github.com/Nawattakorn/ThaiSignLaTranslator.git
   cd ThaiSignLaTranslator
   ```
2. **สร้าง virtual environment** (แนะนำ) และติดตั้ง dependencies
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
3. **รันแอป**
   ```bash
   python app.py  
   ```
   เปิดเว็บเบราว์เซอร์ที่ `http://localhost:5000` (หรือพอร์ตที่แสดงในคอนโซล)

### การอ้างอิง
- แนวทางและชุดข้อมูล : *“Thai Hand Sign Detection using MediaPipe”* – medium.com/@aruchakhem.
---

**หมายเหตุ**: โปรเจคนี้เป็นส่วนหนึ่งของวิชา Machine Learning 
