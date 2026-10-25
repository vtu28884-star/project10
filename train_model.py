import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import xgboost as xgb
import joblib

def load_and_preprocess_dataset(csv_path=None):
    # 1. Locate the entire dataset
    if csv_path is None:
        if os.path.exists("cleaned_vital_signs.csv"):
            csv_path = "cleaned_vital_signs.csv"
        elif os.path.exists("vital_signs_dataset.csv"):
            csv_path = "vital_signs_dataset.csv"
        else:
            csv_candidates = glob.glob("*.csv")
            if not csv_candidates:
                raise FileNotFoundError("No CSV dataset found in working directory. Please provide vital_signs_dataset.csv.")
            csv_path = csv_candidates[0]

    print(f"[+] Loading entire dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[+] Ingested total: {len(df):,} rows, {len(df.columns)} columns.")

    # 2. Standardize column headers
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    print("[+] Columns detected:", df.columns.tolist())

    # Map naming variations to standard clinical names
    col_mapping = {
        'heartrate': 'heart_rate',
        'pulse': 'heart_rate',
        'oxygen': 'spo2',
        'oximeter': 'spo2',
        'oxygen_saturation': 'spo2',
        'temperature': 'temp_c',
        'temp': 'temp_c',
        'body_temp': 'temp_c',
        'body_temperature': 'temp_c',
        'sex': 'gender',
        'height_cm': 'height',
        'height_(m)': 'height_m',
        'weight_kg': 'weight',
        'weight_(kg)': 'weight'
    }
    df.rename(columns=col_mapping, inplace=True)
    # Deduplicate columns if any mapping created duplicates
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 3. Handle gender encoding (M/Male/1: 1, F/Female/0: 0)
    if 'gender' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['gender']):
            gender_series = df['gender'].astype(str).str.strip().str.upper()
            gender_map = {
                'M': 1, 'MALE': 1, '1': 1, '1.0': 1,
                'F': 0, 'FEMALE': 0, '0': 0, '0.0': 0
            }
            df['gender'] = gender_series.map(gender_map)
        mode_val = 1
        valid_modes = df['gender'].dropna()
        if not valid_modes.empty:
            mode_val = valid_modes.mode()[0]
        df['gender'] = pd.to_numeric(df['gender'].fillna(mode_val), errors='coerce').fillna(mode_val).astype(int)

    # 4. Standardize temperature (Convert Fahrenheit to Celsius if mean > 50)
    if 'temp_c' in df.columns:
        if df['temp_c'].mean() > 50.0:
            print("[+] Detected body temperature in Fahrenheit. Converting to Celsius...")
            df['temp_c'] = (df['temp_c'] - 32.0) * (5.0 / 9.0)
        df['temp_c'] = df['temp_c'].round(1)

    # 5. Standardize height (convert meters to cm if mean < 3.0)
    if 'height_m' in df.columns and 'height' not in df.columns:
        df['height'] = df['height_m'] * 100.0
    elif 'height' in df.columns:
        if df['height'].mean() < 3.0:
            df['height'] = df['height'] * 100.0
        df['height'] = df['height'].round(1)

    # 6. Calculate or align BMI
    if 'bmi' in df.columns:
        if isinstance(df['bmi'], pd.DataFrame):
            df['bmi'] = df['bmi'].iloc[:, 0]
    elif 'derived_bmi' in df.columns:
        df['bmi'] = df['derived_bmi']
    elif 'height' in df.columns and 'weight' in df.columns:
        h_m = df['height'] / 100.0 if df['height'].mean() > 3.0 else df['height']
        df['bmi'] = (df['weight'] / (h_m ** 2)).round(2)
    else:
        df['bmi'] = 22.5

    # Deduplicate columns
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 7. Ensure all primary features exist; impute baseline if absent
    required_features = ['age', 'gender', 'height', 'bmi', 'heart_rate', 'spo2', 'temp_c']
    for feat in required_features:
        if feat not in df.columns:
            print(f"[!] Warning: Missing '{feat}'. Imputing physiological baseline.")
            if feat == 'age': df[feat] = 25
            elif feat == 'gender': df[feat] = 1
            elif feat == 'height': df[feat] = 170.0
            elif feat == 'heart_rate': df[feat] = 75.0
            elif feat == 'spo2': df[feat] = 98.0
            elif feat == 'temp_c': df[feat] = 36.8

    # 8. Standardize target classification labels
    label_names_map = {}
    target_candidates = ['risk_category', 'risk_level', 'status', 'target_code', 'target_label', 'target', 'condition', 'risk']
    found_target = None
    for cand in target_candidates:
        matched = [col for col in df.columns if (cand == col or cand in col) and col not in required_features]
        if matched:
            found_target = matched[0]
            break

    if found_target:
        print(f"[+] Found target column: '{found_target}'")
        if not pd.api.types.is_numeric_dtype(df[found_target]):
            # Clean string labels
            raw_labels = df[found_target].astype(str).str.strip().str.title()
            categories = sorted(raw_labels.unique())
            label_map = {cat: idx for idx, cat in enumerate(categories)}
            label_names_map = {idx: f"{cat} ({idx})" for cat, idx in label_map.items()}
            print(f"[+] Target class mapping: {label_map}")
            df['target_label'] = raw_labels.map(label_map)
        else:
            df['target_label'] = df[found_target].astype(int)
            unique_classes = sorted(df['target_label'].unique())
            default_labels = {0: "Normal (0)", 1: "Pyrexia / Fever (1)", 2: "Tachy/Bradycardia (2)", 3: "Hypoxia (3)"}
            label_names_map = {c: default_labels.get(c, f"Class {c}") for c in unique_classes}
    else:
        print("[+] Synthesizing multi-condition clinical target labels from physiological parameters...")
        conditions = [
            df['spo2'] < 92,
            df['temp_c'] > 38.3,
            (df['heart_rate'] > 105) | (df['heart_rate'] < 50)
        ]
        choices = [3, 1, 2]  # 3: Hypoxia, 1: High Fever, 2: Tachycardia/Bradycardia
        df['target_label'] = np.select(conditions, choices, default=0)
        default_labels = {0: "Normal (0)", 1: "Pyrexia / Fever (1)", 2: "Tachy/Bradycardia (2)", 3: "Hypoxia (3)"}
        label_names_map = default_labels

    # Verify target label has at least 2 distinct classes
    unique_targets = sorted(df['target_label'].dropna().unique())
    if len(unique_targets) < 2:
        print("[!] Target column contains fewer than 2 distinct classes. Applying physiological risk stratification...")
        # Stratify into Low Risk (0) vs High Risk (1) using clinical variance
        hr_elevated = (df['heart_rate'] > df['heart_rate'].median()) | (df['spo2'] < df['spo2'].median())
        df['target_label'] = hr_elevated.astype(int)
        label_names_map = {0: "Low Risk (0)", 1: "High Risk (1)"}

    # Clean missing / invalid values and deduplicate columns strictly
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    df = df[required_features + ['target_label']].dropna()
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    X = df[required_features]
    if isinstance(X, pd.DataFrame):
        X = X.loc[:, ~X.columns.duplicated(keep='first')]

    y = df['target_label']
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = y.astype(int)

    print(f"[+] Total cleaned samples ready for training & validation: {len(df):,}")
    print(f"[+] Verified feature columns: {X.columns.tolist()}")

    return X, y, label_names_map

def train():
    # 1. Ingest entire dataset
    X, y, label_names_map = load_and_preprocess_dataset()
    total_samples = len(X)
    print(f"\n[+] Total dataset: {total_samples:,} samples with {X.shape[1]} physiological features.")
    print("[+] Class distribution across entire dataset:\n", y.value_counts())

    # 2. 80% for training and 20% for validation (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\n[+] Dataset Partitioning (80% Train / 20% Validation):")
    print(f"    ► Training Set (80%):   {len(X_train):,} samples ({(len(X_train)/total_samples)*100:.1f}%)")
    print(f"    ► Validation Set (20%): {len(X_val):,} samples ({(len(X_val)/total_samples)*100:.1f}%)")

    # 3. Determine unique classes and configure XGBoost
    classes = sorted(np.unique(y_train))
    num_classes = len(classes)
    print(f"[+] Number of distinct diagnostic classes: {num_classes} -> {classes}")

    if num_classes == 2:
        # Binary Classification
        model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    else:
        # Multi-Class Classification
        model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=num_classes,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1
        )

    print("\n[+] Training XGBoost model on 80% partition with validation tracking...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )

    # 4. Predict on the 20% Validation Set
    y_val_pred = model.predict(X_val)

    # 5. Calculate Macro F1 Score & Validation Metrics
    macro_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_val, y_val_pred)

    print("\n" + "=" * 65)
    print("  MODEL EVALUATION METRICS (20% UNSEEN VALIDATION SET)")
    print("=" * 65)
    print(f"  ► MACRO F1-SCORE:       {macro_f1:.4f}  ({macro_f1 * 100:.2f}%)")
    print(f"  ► WEIGHTED F1-SCORE:    {weighted_f1:.4f}  ({weighted_f1 * 100:.2f}%)")
    print(f"  ► ACCURACY:             {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print("=" * 65)

    # Resolve present classes for classification_report
    present_classes = sorted(list(set(y_val).union(set(y_val_pred))))
    target_names = [label_names_map.get(c, f"Class {c}") for c in present_classes]

    print("\n--- Detailed Validation Classification Report ---")
    print(classification_report(y_val, y_val_pred, labels=present_classes, target_names=target_names, digits=4, zero_division=0))

    print("--- Confusion Matrix (Validation 20%) ---")
    print(confusion_matrix(y_val, y_val_pred, labels=present_classes))

    # 6. Export Model Artifact
    model_path = "wban_xgboost_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n[✓] Model successfully serialized and exported to '{model_path}'")

    return model, macro_f1

if __name__ == "__main__":
    train()