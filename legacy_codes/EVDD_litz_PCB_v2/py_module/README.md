# One-by-one Regression Training

This script trains one regression model per output column from `output_data.csv`, using the following as inputs:

```
Ltx Lrx1 Lrx2 M1 M2 k1 k2 Lmt Lmr1 Lmr2 Llt Llr1 Llr2 Rtx Rrx1 Rrx2 
copperloss_Tx copperloss_Rx1 copperloss_Rx2 coreloss 
B_core B_left B_right B_center B_top_left B_bottom_left B_top_right B_bottom_right 
magnetizing_copperloss_Tx magnetizing_copperloss_Rx1 magnetizing_copperloss_Rx2
```

All other columns are treated as separate targets, and a regressor is trained for each.

## Setup

- Python 3.9+
- Install dependencies:

```
pip install -r py_module/requirements.txt
```

## Run

```
python py_module/train_regressors.py --csv output_data.csv --out models
```

Outputs:
- Saved models: `models/<target>.joblib`
- Metrics summary: `models/metrics.csv`

The script holds out 20% for testing and reports R2/MAE/RMSE per target.
