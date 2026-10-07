# Evaluation results

## webcam_Teerepat_test_base (n=72)

| model | accuracy | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|
| KNN + DTW | 76.39% | 86.11% | 0.735 | 6.7 |
| CNN | 81.94% | 93.06% | 0.811 | 1.3 |
| GRU | 79.17% | 87.50% | 0.769 | 5.5 |
| Ensemble (เฉลี่ย 3 ตัว) | 86.11% | 94.44% | 0.846 | 13.5 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 0→O: 2, 8→V: 2, 9→4: 2, 2→Z: 1, 5→0: 1, E→O: 1, E→P: 1, G→E: 1
- **CNN**: 0→O: 2, O→0: 2, 6→5: 1, 7→9: 1, 9→Q: 1, G→9: 1, M→H: 1, P→X: 1
- **GRU**: 0→O: 2, 8→E: 2, 9→4: 2, 5→P: 1, 7→9: 1, G→E: 1, M→A: 1, N→V: 1
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 2, 9→4: 2, 5→0: 1, G→E: 1, N→V: 1, P→X: 1, X→9: 1, X→V: 1
