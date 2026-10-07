# Evaluation results

## standard_order_tuned_std (n=360)

| model | accuracy | within digits / letters | top-3 accuracy | macro-F1 | latency (median ms, CPU) |
|---|---|---|---|---|---|
| KNN + DTW | 73.61% | 77.78% | 82.22% | 0.670 | 12.3 |
| CNN | 86.67% | 93.06% | 97.22% | 0.861 | 6.4 |
| GRU | 77.78% | 80.56% | 83.33% | 0.713 | 11.8 |
| Ensemble (เฉลี่ย 3 ตัว) | 80.56% | 83.33% | 93.06% | 0.752 | 32.6 |

Most common mistakes (true → predicted: count):

- **KNN + DTW**: 0→O: 10, A→H: 10, G→O: 10, I→B: 10, J→0: 10, T→P: 10, E→R: 9, 9→S: 6
- **CNN**: 0→O: 10, O→0: 9, J→O: 6, 8→6: 3, E→F: 3, G→C: 3, J→U: 3, 4→A: 2
- **GRU**: 0→O: 10, 8→5: 10, 9→4: 10, J→0: 10, T→P: 10, G→O: 7, E→B: 6, I→R: 6
- **Ensemble (เฉลี่ย 3 ตัว)**: 0→O: 10, I→B: 10, J→0: 10, T→P: 10, 9→4: 9, G→O: 9, E→R: 8, 9→S: 1
