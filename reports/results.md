| Model | Accuracy | Macro F1 | Weighted F1 | Trainable params | Train time (min) |
|---|---|---|---|---|---|
| LSTM | 0.7714 | 0.6807 | 0.7750 | 1,091,034 | 0.7 |
| GRU | 0.7751 | 0.6817 | 0.7745 | 1,061,594 | 0.5 |
| BiLSTM + Attention | 0.7782 | 0.6796 | 0.7797 | 1,242,586 | 1.3 |
| DistilBERT | 0.8141 | 0.7079 | 0.8148 | 66,958,086 | 38.7 |

Per-class F1 (test):

| Model | joy | sadness | anger | fear | surprise | disgust |
|---|---|---|---|---|---|---|
| LSTM | 0.873 | 0.630 | 0.644 | 0.682 | 0.697 | 0.560 |
| GRU | 0.874 | 0.629 | 0.634 | 0.686 | 0.700 | 0.566 |
| BiLSTM + Attention | 0.877 | 0.628 | 0.657 | 0.655 | 0.705 | 0.556 |
| DistilBERT | 0.908 | 0.660 | 0.706 | 0.692 | 0.745 | 0.536 |
