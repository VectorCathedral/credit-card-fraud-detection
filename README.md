# Credit Card Fraud Detection

## Overview
This repository contains a Jupyter Notebook, **cc_fraud_detection.ipynb**, that builds and evaluates multiple models to detect fraudulent credit card transactions using the public **Kaggle ULB Credit Card Fraud Detection** dataset (`mlg-ulb/creditcardfraud`).

The workflow covers:
- Basic EDA and class distribution checks.
- Feature scaling for `Amount` (RobustScaler) and min–max normalization for `Time`.
- Deterministic train/test/validation splits by row index.
- A baseline set of classical ML models.
- Simple Keras-based neural networks.
- A second experiment that **balances** the dataset via undersampling for comparison.

## Data
- **Source:** Kaggle – mlg-ulb/creditcardfraud (European card transactions, 284,807 rows, anonymized PCA features V1–V28, `Time`, `Amount`, and target `Class`).
- **Access in notebook:**  
  ```python
  import kagglehub
  path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
  df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
  ```
  > If you are not running inside Kaggle, replace the `read_csv` path with the `path` returned by `kagglehub` (or download the CSV and point to it locally).

## EDA & Preprocessing
- `df.Class.value_counts()` to inspect imbalance.
- `df.hist(...)` for quick feature histograms.
- **Scaling**
  - `Amount` → `RobustScaler` (less sensitive to outliers).
  - `Time` → min–max normalization to `[0, 1]`.
- **Splits (by contiguous index ranges):**
  - `train = new_df[:240000]`
  - `test  = new_df[240000:262000]`
  - `val   = new_df[262000:]`
- **Balanced experiment:**  
  Create `balanced_df` by concatenating all fraud rows with an equal number of randomly sampled non-fraud rows (`random_state=1`). Then split into `train_b`, `test_b`, `val_b` using the same index cutoffs.

## Feature/Target Tensors
- For each split, features are all columns except `Class`; targets are `Class`.
- Arrays: `x_train_np`, `y_train_np`, `x_test_np`, `y_test_np`, `x_val_np`, `y_val_np` (and the `_b` variants for the balanced subset).

## Models
### Classical ML (scikit-learn)
- **LogisticRegression()** — default parameters.
- **RandomForestClassifier(max_depth=2, n_jobs=-1)**
- **GradientBoostingClassifier(...)**
  - Imbalanced run: `n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0`
  - Balanced run:   `n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0`
- **LinearSVC(class_weight='balanced')`**

### Neural Networks (TensorFlow/Keras)
Helper:
```python
def neural_net_predictions(model, x):
    return (model.predict(x).flatten() > 0.5).astype(int)
```

- **shallow_nn** (imbalanced data)
  - InputLayer(shape = number of features)
  - Dense(16, activation='relu') → BatchNormalization → Dropout(0.3)
  - Dense(8,  activation='relu') → BatchNormalization → Dropout(0.3)
  - Dense(1,  activation='sigmoid')
  - Compile: `optimizer='adam'`, `loss='binary_crossentropy'`, `metrics=['accuracy']`
  - Training: `epochs=10`, with `ModelCheckpoint('shallow_nn.keras', save_best_only=True)`

- **shallow_nn_b** (balanced data)
  - InputLayer → Dense(2, 'relu') → BatchNormalization → Dense(1, 'sigmoid')
  - Compile: Adam + BCE; `epochs=40`
  - Checkpoint: `shallow_nn_b.keras`

- **shallow_nn_b1** (balanced data, variant)
  - InputLayer → Dense(1, 'relu') → BatchNormalization → Dense(1, 'sigmoid')
  - Compile: Adam + BCE; `epochs=40`
  - Checkpoint: `shallow_nn_b1.keras`

## Evaluation
- For all models, the notebook prints `classification_report` on the **validation set** (`val` or `val_b`), which includes **precision, recall, f1-score, and support** for:
  - `Not Fraud` (Class 0)
  - `Fraud` (Class 1)

> Note: No ROC/PR curves are produced in the current notebook; metrics are report-based only.

## How to Run
1. Install dependencies (see below).
2. Ensure the Kaggle dataset is accessible. If not using Kaggle:
   - Use `kagglehub.dataset_download("mlg-ulb/creditcardfraud")` and update the `read_csv` path accordingly, **or**
   - Download the CSV manually and set the path in `pd.read_csv(...)`.
3. Open the notebook and run cells in order.

## Dependencies
- `pandas`
- `scikit-learn`
- `tensorflow` (Keras API)
- `kagglehub`
- (optional) `matplotlib` for plotting via pandas

### Quick install
```bash
pip install pandas scikit-learn tensorflow kagglehub matplotlib
```

## Artifacts
Training will save the following Keras model files (best checkpoints) in the working directory:
- `shallow_nn.keras`
- `shallow_nn_b.keras`
- `shallow_nn_b1.keras`

## Project Structure
```
cc_fraud_detection.ipynb
```

## Notes & Limitations
- Splits are **by row index**, not stratified/random; results may be sensitive to ordering.
- Class imbalance in the original data is severe; the balanced experiment uses **undersampling** (no SMOTE).
- LinearSVC is run with `class_weight='balanced'` to partially mitigate imbalance.

## License
This project is provided for educational purposes. Check Kaggle’s dataset license for usage constraints.
