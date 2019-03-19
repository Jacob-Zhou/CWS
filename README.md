# CWS

Chinese word segmentation model

## Results

### 2019.3.19

#### 72 buckets, viterbi decoding, <img src="https://latex.codecogs.com/gif.latex?ffn(x_{lstm} \oplus x_{span} \oplus e_{indepedent-char})"/>

|       | Precision | Recall | F-score | accuracy |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :------: | :-----: | :----: |
| ctb6  |  95.814   | 95.832 | 95.823  |  96.514  | 39.672  |   40   |
|  pku  |  95.954   | 94.596 | 95.270  |  95.914  | 105.281 |   28   |
|  msr  |  97.417   | 97.039 | 97.228  |  97.650  | 161.299 |   55   |

#### 72 buckets, viterbi decoding, <img src="https://latex.codecogs.com/gif.latex?ffn(x_{lstm} \oplus x_{span} \oplus e_{char})"/>

|       | Precision | Recall | F-score | accuracy |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :------: | :-----: | :----: |
| ctb6  |  95.856   | 95.910 | 95.883  |  96.568  | 39.740  |   35   |
|  pku  |  95.487   | 96.643 | 96.061  |  96.630  | 103.781 |   40   |
|  msr  |  96.691   | 97.731 | 97.208  |  97.601  | 159.290 |   48   |

#### 72 buckets, viterbi decoding, <img src="https://latex.codecogs.com/gif.latex?ffn(x_{lstm} \oplus x_{span})"/>

|       | Precision | Recall | F-score | accuracy |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :------: | :-----: | :----: |
| ctb6  |  95.333   | 96.059 | 95.695  |  96.392  | 42.053  |   33   |
|  pku  |  95.701   | 96.614 | 96.155  |  96.720  | 104.353 |   46   |
|  msr  |  96.896   | 97.611 | 97.252  |  97.642  | 159.665 |   67   |

#### 72 buckets, viterbi decoding, <img src="https://latex.codecogs.com/gif.latex?ffn(x_{lstm}) + ffn(x_{span})"/>


|       | Precision | Recall | F-score | accuracy |  Time   | Epochs |
| :---: | :-------: | :----: | :-----: | :------: | :-----: | :----: |
| ctb6  |  95.157   | 95.994 | 95.573  |  96.281  | 42.162  |   37   |
|  pku  |  95.582   | 96.567 | 96.072  |  96.644  | 103.599 |   35   |
|  msr  |  96.815   | 97.699 | 97.255  |  97.650  | 154.205 |   42   |

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
