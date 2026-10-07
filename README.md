# AirWrite — เขียนตัวอักษรกลางอากาศด้วยนิ้ว

เขียนตัวเลข 0–9 และตัวอักษร A–Z กลางอากาศหน้าเว็บแคม แล้วระบบแปลงเป็นข้อความ
โดยเปรียบเทียบโมเดล 3 แบบ: **KNN + DTW**, **CNN** (มองเส้นเป็นภาพ) และ **GRU** (มองเส้นเป็นลำดับเวลา)

```
เว็บแคม → MediaPipe Hands (เบราว์เซอร์) → เส้นทางปลายนิ้ว → POST /api/predict
        → resample 64 จุด + normalize → KNN+DTW / CNN / GRU → ตัวอักษร
```

## ติดตั้ง

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

ต้องมี dataset RTD และ RTC วางไว้ที่ `data/external/rtd/RTD Dataset/` และ `data/external/rtc/`
(ดาวน์โหลดจาก <https://shahinur.com/en/rtd/> และ <https://shahinur.com/en/rtc/>)

## ใช้งาน

```bash
.venv/bin/python scripts/train.py                  # เทรนทั้ง 3 โมเดล (~5 นาทีบน GPU)
.venv/bin/uvicorn server.app:app --port 8000       # เปิด http://127.0.0.1:8000
.venv/bin/python scripts/evaluate.py --writer tpaof  # ประเมินผล -> reports/
```

- `/` (หรือ `/practice`) — หน้าหลัก ฝึกเขียนสำหรับเด็ก: สุ่มตัวให้เขียน ตัวช่วยลากตาม 3 ระดับ บอกถูก/ผิด นับคะแนน และฝึกซ้ำเฉพาะตัวที่ผิด
  มีโหมดฝึกเขียนคำศัพท์ 3–4 ตัวอักษรพร้อมภาพและคำแปลไทย เขียนทีละตัว ตัวที่ผิดต้องเขียนใหม่จนครบคำ
  (เส้นตัวช่วยตามลำดับการลากเส้นมาตรฐาน Zaner-Bloser จาก `airwrite/stroke_order.py`: `scripts/make_guides.py` → `checkpoints/guides.json`)
- `/demo` — เขียนอิสระ ดูผลทั้ง 3 โมเดลข้างกัน และพิมพ์เป็นข้อความทีละตัว (ความสามารถเสริม: ป้อนข้อความแบบไม่สัมผัส)
- `/collect` — เก็บลายมือของผู้ใช้ไว้ทดสอบ/เทรนเพิ่ม (บันทึกที่ `data/collected/<writer>/`)
- `train.py --extra tpaof` — เทรนใหม่โดยเพิ่มลายมือที่เก็บเอง
- `train.py --standard 100` — เพิ่มตัวอย่างสังเคราะห์ตามลำดับเส้นมาตรฐาน (ให้อ่านตัวที่ลากตามเส้นตัวช่วยถูก)
  โมเดลปัจจุบัน: `train.py --extra Teerepat --extra-test-per-class 2 --standard 100`

ท่ามือ: 🤏 จีบนิ้วโป้งกับนิ้วชี้ = เขียน · ปล่อย = ยกปากกา ·
ยกปากกาหรือค้างนิ่ง ~0.8 วินาที = ส่งทาย ·
✋ แบมือค้าง 0.7 วินาที = ยกเลิกตัวที่กำลังเขียน หรือ (ถ้ากระดานว่าง) ลบตัวอักษร (ค้างต่อ = ลบทุก 0.5 วินาที; หน้าเก็บข้อมูล = ลบตัวอย่างล่าสุด)

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `airwrite/preprocess.py` | ต่อเส้น, resample, smooth, normalize, แปลงเป็นลำดับ/ภาพ, augmentation |
| `airwrite/datasets.py` | โหลด RTD/RTC (pickle แบบอนุญาตเฉพาะ numpy) และข้อมูลที่เก็บเอง |
| `airwrite/models.py` | CharCNN, CharGRU, KnnDtw (+ DTW) |
| `airwrite/predictor.py` | โหลดโมเดลแล้วทายพร้อมกันทั้ง 3 ตัว + Ensemble |
| `airwrite/stroke_order.py` | ลำดับเส้นมาตรฐาน (Zaner-Bloser) สำหรับเส้นตัวช่วยและข้อมูลสังเคราะห์ |
| `scripts/train.py`, `scripts/evaluate.py` | เทรน / ประเมินผล + กราฟสำหรับรายงาน |
| `server/app.py` | FastAPI: หน้าเว็บ, `/api/predict`, `/api/guides`, `/api/samples` |
| `web/js/airpen.js` | กล้อง + MediaPipe + ตรวจท่ามือ + One Euro filter + สร้างเส้น |

## เครดิตและแหล่งอ้างอิง

- **RTD / RTC dataset** — M. S. Alam, K.-C. Kwon, M. A. Alam, M. Y. Abbass, S. M. Imtiaz, N. Kim,
  "Trajectory-Based Air-Writing Recognition Using Deep Neural Network and Depth Sensor,"
  *Sensors*, 20(2):376, 2020. doi:10.3390/s20020376
- **MediaPipe Hand Landmarker** (Google) — `web/vendor/` เป็นสำเนาของ `@mediapipe/tasks-vision` 1.1.0 และโมเดล `hand_landmarker.task` (Apache-2.0) เพื่อให้เดโมทำงานได้แบบออฟไลน์
- **ลำดับการลากเส้นของเส้นตัวช่วย** — Zaner-Bloser Handwriting, Manuscript Stroke Descriptions
- **One Euro Filter** — G. Casiez, N. Roussel, D. Vogel, "1€ Filter," CHI 2012
- แนวคิดการใช้ท่ามือ (จีบนิ้วเขียน) และการ resample ตามความยาวเส้น ได้แรงบันดาลใจจาก
  [SkyInk](https://github.com/TheekshanaChathuranga/SkyInk) (MIT) — โค้ดในโปรเจกต์นี้เขียนใหม่ทั้งหมด
