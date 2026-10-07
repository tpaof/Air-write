# Evaluation results

## webcam_Teerepat_base_all (n=180)

| model | accuracy | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|
| KNN + DTW | 75.00% | 84.44% | 0.717 | 6.7 |
| CNN | 83.33% | 93.33% | 0.830 | 1.4 |
| GRU | 82.22% | 87.22% | 0.794 | 5.6 |
| Ensemble (เฉลี่ย 3 ตัว) | 85.00% | 94.44% | 0.828 | 13.7 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 0→O: 5, 8→V: 4, 2→Z: 3, G→E: 3, X→V: 3, 9→Y: 2, 9→4: 2, E→O: 2
- **CNN**: 0→O: 5, O→0: 5, 7→9: 2, X→K: 2, 1→Y: 1, 5→U: 1, 6→B: 1, 6→5: 1
- **GRU**: 0→O: 5, 9→4: 5, 8→E: 4, X→V: 3, 1→Y: 2, 5→F: 1, 5→Y: 1, 5→P: 1
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 5, G→E: 3, X→V: 3, 1→Y: 2, 9→Y: 2, 9→4: 2, 4→Q: 1, 5→P: 1
