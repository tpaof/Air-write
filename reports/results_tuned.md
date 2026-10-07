# Evaluation results

## public_test_tuned (n=3011)

| model | accuracy | within digits / letters | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|---|
| KNN + DTW | 98.70% | 99.50% | 99.63% | 0.988 | 7.8 |
| CNN | 97.94% | 99.67% | 99.70% | 0.981 | 2.0 |
| GRU | 99.70% | 99.70% | 99.77% | 0.997 | 6.2 |
| Ensemble (เฉลี่ย 3 ตัว) | 99.63% | 99.67% | 99.73% | 0.997 | 16.5 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 5→S: 17, O→0: 6, A→H: 2, 6→7: 1, 9→0: 1, 0→1: 1, 1→2: 1, 2→3: 1
- **CNN**: 0→O: 37, O→0: 13, 5→S: 2, 6→7: 1, 9→O: 1, 0→1: 1, 1→2: 1, 2→3: 1
- **GRU**: 6→7: 1, 9→0: 1, 0→1: 1, 1→2: 1, 2→3: 1, 3→4: 1, 4→5: 1, N→M: 1
- **Ensemble (เฉลี่ย 3 ตัว)**: 5→S: 1, 6→7: 1, 9→0: 1, 0→1: 1, 1→2: 1, 2→3: 1, 3→4: 1, 4→5: 1

## webcam_Teerepat_test_tuned (n=72)

| model | accuracy | within digits / letters | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|---|
| KNN + DTW | 93.06% | 94.44% | 97.22% | 0.923 | 7.7 |
| CNN | 94.44% | 97.22% | 98.61% | 0.934 | 1.8 |
| GRU | 94.44% | 97.22% | 97.22% | 0.934 | 5.9 |
| Ensemble (เฉลี่ย 3 ตัว) | 94.44% | 97.22% | 97.22% | 0.934 | 15.7 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: E→P: 1, P→X: 1, S→5: 1, X→7: 1, X→V: 1
- **CNN**: 0→O: 2, P→X: 1, X→F: 1
- **GRU**: 0→O: 2, P→X: 1, X→7: 1
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 2, P→X: 1, X→7: 1

## standard_order_tuned (n=360)

| model | accuracy | within digits / letters | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|---|
| KNN + DTW | 98.89% | 100.00% | 100.00% | 0.988 | 7.9 |
| CNN | 95.56% | 99.44% | 100.00% | 0.954 | 2.0 |
| GRU | 97.22% | 100.00% | 100.00% | 0.963 | 6.1 |
| Ensemble (เฉลี่ย 3 ตัว) | 97.78% | 100.00% | 100.00% | 0.974 | 16.5 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 0→O: 4
- **CNN**: 0→O: 9, O→0: 5, P→R: 2
- **GRU**: 0→O: 10
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 8
