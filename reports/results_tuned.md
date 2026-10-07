# Evaluation results

## public_test_tuned (n=3011)

| model | accuracy | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|
| KNN + DTW | 98.87% | 99.63% | 0.989 | 7.1 |
| CNN | 98.54% | 99.77% | 0.986 | 1.6 |
| GRU | 99.70% | 99.73% | 0.997 | 5.7 |
| Ensemble (เฉลี่ย 3 ตัว) | 99.63% | 99.77% | 0.997 | 14.5 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 5→S: 17, A→H: 3, 6→7: 1, 9→0: 1, 0→1: 1, 1→2: 1, 2→3: 1, 3→4: 1
- **CNN**: O→0: 20, 0→O: 5, 5→S: 4, 6→7: 1, 0→U: 1, 9→0: 1, 0→1: 1, 1→2: 1
- **GRU**: 6→7: 1, 9→0: 1, 0→1: 1, 1→2: 1, 2→3: 1, 3→4: 1, 4→5: 1, N→M: 1
- **Ensemble (เฉลี่ย 3 ตัว)**: 5→S: 1, 6→7: 1, 9→0: 1, 0→1: 1, 1→2: 1, 2→3: 1, 3→4: 1, 4→5: 1

## webcam_Teerepat_test_tuned (n=72)

| model | accuracy | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|
| KNN + DTW | 90.28% | 97.22% | 0.895 | 7.0 |
| CNN | 86.11% | 97.22% | 0.852 | 1.5 |
| GRU | 94.44% | 97.22% | 0.934 | 5.5 |
| Ensemble (เฉลี่ย 3 ตัว) | 94.44% | 97.22% | 0.934 | 14.1 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 0→O: 1, 5→0: 1, E→P: 1, P→X: 1, S→5: 1, X→7: 1, X→V: 1
- **CNN**: 0→O: 2, O→0: 2, 9→S: 1, B→8: 1, M→H: 1, P→X: 1, X→F: 1, X→Y: 1
- **GRU**: 0→O: 2, P→X: 1, X→9: 1
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 2, P→X: 1, X→7: 1
