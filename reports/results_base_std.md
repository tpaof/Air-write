# Evaluation results

## standard_order_base_std (n=360)

| model | accuracy | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|
| KNN + DTW | 68.06% | 72.50% | 0.614 | 6.4 |
| CNN | 86.11% | 96.39% | 0.850 | 1.5 |
| GRU | 75.00% | 77.78% | 0.698 | 5.3 |
| Ensemble (เฉลี่ย 3 ตัว) | 76.11% | 90.56% | 0.711 | 13.2 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 0→O: 10, 8→E: 10, A→H: 10, G→O: 10, I→B: 10, J→0: 10, T→P: 10, E→R: 8
- **CNN**: E→F: 10, O→0: 9, 0→O: 8, J→O: 7, T→V: 6, Q→O: 3, J→U: 2, Q→0: 2
- **GRU**: 0→O: 10, 5→0: 10, 9→4: 10, E→R: 10, I→R: 10, J→0: 10, G→6: 9, 8→E: 8
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 10, G→O: 10, I→B: 10, J→0: 10, T→P: 10, 9→4: 9, E→R: 8, 5→0: 5
