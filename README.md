# CWS

Chinese word segmentation model

## Results

### 2019.3.16

#### No buckets, viterbi decoding

|       | Precision | Recall | F-score |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :-----: | :----: |
| ctb6  |  96.011   | 95.780 | 95.895  | 145.296 |   61   |
|  pku  |  96.363   | 96.499 | 96.431  | 310.249 |   69   |
|  msr  |  97.151   | 97.677 | 97.413  | 514.783 |   57   |

#### No buckets

|       | Precision | Recall | F-score |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :-----: | :----: |
| ctb6  |  95.957   | 95.927 | 95.942  | 84.886  |   57   |
|  pku  |  96.273   | 96.306 | 96.289  | 201.262 |   46   |
|  msr  |  97.202   | 97.551 | 97.376  | 303.233 |   50   |

#### 72 buckets, viterbi decoding

|       | Precision | Recall | F-score |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :-----: | :----: |
| ctb6  |  95.839   | 95.768 | 95.803  | 103.079 |   50   |
|  pku  |  96.280   | 96.483 | 96.381  | 216.211 |   54   |
|  msr  |  97.350   | 97.331 | 97.340  | 356.368 |   40   |

#### 72 buckets

|       | Precision | Recall | F-score |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :-----: | :----: |
| ctb6  |  95.328   | 95.370 | 95.349  | 38.748  |   68   |
|  pku  |  96.064   | 96.592 | 96.327  | 100.834 |   40   |
|  msr  |  97.323   | 97.446 | 97.384  | 147.579 |   58   |
