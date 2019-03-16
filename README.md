# CWS

Chinese word segmentation model

## Results

### 2019.3.16

#### No buckets, viterbi decoding



|       | Precision | Recall | F-score | Epochs |  Time   |
| :---: | :-------: | :----: | :-----: | :----: | :-----: |
| ctb6  |  96.011   | 95.780 | 95.895  |   61   | 145.296 |
|  pku  |  96.363   | 96.499 | 96.431  |   69   | 310.249 |
|  msr  |  97.151   | 97.677 | 97.413  |   57   | 514.783 |

#### 72 buckets, viterbi decoding





|       | Precision | Recall | F-score | Epochs |  Time   |
| :---: | :-------: | :----: | :-----: | :----: | :-----: |
| ctb6  |  95.839   | 95.768 | 95.803  |   50   | 103.079 |
|  pku  |  96.280   | 96.483 | 96.381  |   54   | 216.211 |
|  msr  |  97.350   | 97.331 | 97.340  |   40   | 356.368 |
