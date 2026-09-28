# Two-Stage Diabetes Detection & Classification using Machine Learning

A final year project that predicts diabetes from clinical and symptom data in **two stages**, and explains each prediction with **LIME**.

- **Stage 1 – Detection:** Non-Diabetic vs Diabetic
- **Stage 2 – Classification:** if diabetic, Diabetic (general) vs Gestational


---

## Features

- Two-stage pipeline (detection → type classification)
- 5 models compared at each stage: Random Forest, SVM, Logistic Regression, Decision Tree, KNN
- Evaluation: confusion matrix, classification report, 5-fold stratified cross-validation (mean ± std)
- Statistical checks: t-test and correlation per feature, McNemar's test between models
- Robustness check across multiple random seeds (0, 42, 70, 300)
- Explainable AI: LIME explanation for every prediction
- Feature importance from the best model
- Trained models exported with `joblib` for use in a backend/app

## Dataset

- **Size:** 2,149 records, 12 input features
- **Classes:** Non-Diabetic, Diabetic, Gestational
- **Features:** Gender, Age, BSR (mg/dl), Systolic, Diastolic, Peripheral Neuropathy, BMI, Delayed Healing, Genetic Relation, Frequent Urination, Dry Mouth, Frequent Hunger
- **Target column:** `Diabetes_Status`

The dataset is **not included** in this repository (patient data). Place your Excel file in the project folder and set `FILE_PATH` in the notebook's configuration cell.

## Results

| Stage | Best model | Test accuracy |
|-------|-----------|---------------|
| 1 – Detection | Random Forest | 100% |
| 2 – Classification | Random Forest | 99.0% |

5-fold CV accuracy (mean): Stage 1 ≈ 99.4–100%, Stage 2 ≈ 96.9–98.8% depending on the model.

Most important feature (Random Forest, Stage 1): **BSR mg/dl** (~59% importance), followed by Frequent Urination, Dry Mouth and Frequent Hunger.

## Project Structure

```
.
├── DiabetesClassification.ipynb   # Full pipeline: training, evaluation, LIME, model export
├── requirements.txt
├── README.md
├── .gitignore
└── (generated after running) det_model.pkl, scaler_det.pkl,
    cls_model.pkl, scaler_cls.pkl, feature_names.pkl, training_sample.pkl
```

## Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
jupyter notebook DiabetesClassification.ipynb
```

Or open the notebook directly in Google Colab and upload the dataset.

## Usage

1. Put the dataset file in the project folder and update `FILE_PATH`.
2. Run all cells in order.
3. The last cells save the trained models (`.pkl`). Load them in your backend:

```python
import joblib
import pandas as pd

det_model  = joblib.load("det_model.pkl")
scaler_det = joblib.load("scaler_det.pkl")
cls_model  = joblib.load("cls_model.pkl")
scaler_cls = joblib.load("scaler_cls.pkl")
features   = joblib.load("feature_names.pkl")

patient = pd.DataFrame([[1, 41, 168, 130, 84, 0, 24.5, 0, 0, 0, 0, 0]], columns=features)

if det_model.predict(scaler_det.transform(patient))[0] == 0:
    print("Non-Diabetic")
else:
    print("Diabetic type:", cls_model.predict(scaler_cls.transform(patient))[0])
```

## Tech Stack

Python, pandas, NumPy, scikit-learn, SciPy, statsmodels, LIME, Matplotlib, Seaborn, joblib

## Author

**Syed Abdullah Shamsi ,Haseeb Raza** – Final Year Project, University of Sialkot,Department of Computing & IT, 2025-2026
Supervisor: Dr Adven

## License

MIT License .
