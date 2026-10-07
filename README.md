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
.venv/bin/uvicorn server.app:app --port 8001       # เปิด http://127.0.0.1:8001
.venv/bin/python scripts/evaluate.py --writer tpaof  # ประเมินผล -> reports/
```

- `/` — หน้าเดโม เขียนแล้วดูผลทั้ง 3 โมเดลข้างกัน
- `/collect` — เก็บลายมือของผู้ใช้ไว้ทดสอบ/เทรนเพิ่ม (บันทึกที่ `data/collected/<writer>/`)
- `train.py --extra tpaof` — เทรนใหม่โดยเพิ่มลายมือที่เก็บเอง

ท่ามือ: 🤏 จีบนิ้วโป้งกับนิ้วชี้ = เขียน · ปล่อย = ยกปากกา ·
ยกปากกาหรือค้างนิ่ง ~0.8 วินาที = ส่งทาย · 🤏 จีบบนคำที่แถบบนของภาพ = เลือกคำที่ระบบเดา ·
✋ แบมือค้าง 0.7 วินาที = ยกเลิกตัวที่กำลังเขียน หรือ (ถ้ากระดานว่าง) ลบตัวอักษร (ค้างต่อ = ลบทุก 0.5 วินาที; หน้าเก็บข้อมูล = ลบตัวอย่างล่าสุด)

## เดาคำที่กำลังเขียน (Bayesian + bigram language model)

หลังเขียนแต่ละตัว `airwrite/lexicon.py` เดาคำที่ผู้ใช้น่าจะกำลังเขียน 3 คำ (เฉพาะคำในพจนานุกรม) แสดงเป็นปุ่มบนแถบบนของภาพ ยกมือไปจีบที่คำเพื่อเลือก:

    P(คำ | ตัวที่เขียนแล้ว, คำก่อนหน้า) ∝ P(คำ | คำก่อนหน้า) · Π P(ตัวที่ i | เส้นที่ i)

- P(คำ | คำก่อนหน้า) = 0.7 · bigram + 0.3 · ความถี่คำ (เช่น THANK → YOU)
- 10% ของ prior ให้ "สตริงอะไรก็ได้" เพื่อให้รหัส/ตัวเลขที่ไม่อยู่ในพจนานุกรมยังพิมพ์ได้
- เริ่มเดาหลังเขียนตัวแรกของคำแล้ว (คำก่อนหน้ายังช่วยจัดอันดับผ่าน bigram)
- ช่วยแก้คู่ที่รูปร่างเหมือนกัน (O/0, I/1, S/5) จากบริบท เช่น `HELL0` → `HELLO`

ไฟล์: `data/lexicon/words_en.tsv` (20,000 คำ), `bigrams_en.tsv`, `custom_words.txt` (เพิ่มคำเองได้ บรรทัดละคำ แล้วรีสตาร์ท server)

## โครงสร้าง

| ไฟล์ | หน้าที่ |
|---|---|
| `airwrite/preprocess.py` | ต่อเส้น, resample, smooth, normalize, แปลงเป็นลำดับ/ภาพ, augmentation |
| `airwrite/datasets.py` | โหลด RTD/RTC (pickle แบบอนุญาตเฉพาะ numpy) และข้อมูลที่เก็บเอง |
| `airwrite/models.py` | CharCNN, CharGRU, KnnDtw (+ DTW) |
| `airwrite/predictor.py` | โหลดโมเดลแล้วทายพร้อมกันทั้ง 3 ตัว + Ensemble |
| `airwrite/lexicon.py` | เดาคำ (Bayesian + bigram) |
| `scripts/train.py`, `scripts/evaluate.py` | เทรน / ประเมินผล + กราฟสำหรับรายงาน |
| `server/app.py` | FastAPI: หน้าเว็บ, `/api/predict`, `/api/suggest`, `/api/samples` |
| `web/js/airpen.js` | กล้อง + MediaPipe + ตรวจท่ามือ + One Euro filter + สร้างเส้น |

## เครดิตและแหล่งอ้างอิง

- **RTD / RTC dataset** — M. S. Alam, K.-C. Kwon, M. A. Alam, M. Y. Abbass, S. M. Imtiaz, N. Kim,
  "Trajectory-Based Air-Writing Recognition Using Deep Neural Network and Depth Sensor,"
  *Sensors*, 20(2):376, 2020. doi:10.3390/s20020376
- **MediaPipe Hand Landmarker** (Google) — `web/vendor/` เป็นสำเนาของ `@mediapipe/tasks-vision` 1.1.0 และโมเดล `hand_landmarker.task` (Apache-2.0) เพื่อให้เดโมทำงานได้แบบออฟไลน์
- **ความถี่คำและ bigram** — `count_1w.txt`, `count_2w.txt` จาก Peter Norvig, "Natural Language Corpus Data" (<https://norvig.com/ngrams/>) ซึ่งมาจาก Google Web Trillion Word Corpus
- **One Euro Filter** — G. Casiez, N. Roussel, D. Vogel, "1€ Filter," CHI 2012
- แนวคิดการใช้ท่ามือ (จีบนิ้วเขียน) และการ resample ตามความยาวเส้น ได้แรงบันดาลใจจาก
  [SkyInk](https://github.com/TheekshanaChathuranga/SkyInk) (MIT) — โค้ดในโปรเจกต์นี้เขียนใหม่ทั้งหมด
