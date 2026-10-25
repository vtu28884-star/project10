import pandas as pd
import numpy as np
import glob
import os

def clean_wban_dataset(input_file="vital_signs_dataset.csv", output_file="cleaned_vital_signs.csv"):
    # 1. Locate file
    if not os.path.exists(input_file):
        csv_candidates = glob.glob("*.csv")
        if not csv_candidates:
            print("[!] Error: No CSV file found in current folder.")
            return
        input_file = csv_candidates[0]
        
    print(f"[+] Loading raw dataset from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"[+] Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Standardize column headers
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    print("\n[+] Initial column names detected:", df.columns.tolist())

    # Map variations of names to standard names
    rename_dict = {
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
        'weight_kg': 'weight'
    }
    df.rename(columns=rename_dict, inplace=True)

    # 3. Check and drop duplicate rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    if duplicates_removed > 0:
        print(f"[+] Removed {duplicates_removed} duplicate rows.")

    # 4. Handle Gender Encoding (Male: 1, Female: 0)
    if 'gender' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['gender']):
            gender_series = df['gender'].astype(str).str.strip().str.upper()
            gender_map = {
                'M': 1, 'MALE': 1, '1': 1, '1.0': 1,
                'F': 0, 'FEMALE': 0, '0': 0, '0.0': 0
            }
            df['gender'] = gender_series.map(gender_map)
        # Fill any missing gender with mode or default to 1 (Male)
        mode_val = 1
        valid_modes = df['gender'].dropna()
        if not valid_modes.empty:
            mode_val = valid_modes.mode()[0]
        df['gender'] = pd.to_numeric(df['gender'].fillna(mode_val), errors='coerce').fillna(mode_val).astype(int)

    # 5. Fix Temperature Units (Fahrenheit to Celsius)
    if 'temp_c' in df.columns:
        # If mean temperature is > 50, it is almost certainly in Fahrenheit
        if df['temp_c'].mean() > 50:
            print("[+] Temperature detected in Fahrenheit. Converting to Celsius...")
            df['temp_c'] = (df['temp_c'] - 32.0) * (5.0 / 9.0)
        df['temp_c'] = df['temp_c'].round(1)

    # 6. Height & Weight Standardization and BMI derivation
    if 'height' in df.columns:
        # If height is in meters (e.g., 1.75), convert to cm
        if df['height'].mean() < 3.0:
            df['height'] = df['height'] * 100.0
        df['height'] = df['height'].round(1)

    if 'bmi' in df.columns:
        if isinstance(df['bmi'], pd.DataFrame):
            df['bmi'] = df['bmi'].iloc[:, 0]
    elif 'derived_bmi' in df.columns:
        df['bmi'] = df['derived_bmi']
    elif 'height' in df.columns and 'weight' in df.columns:
        h_meters = df['height'] / 100.0
        df['bmi'] = (df['weight'] / (h_meters ** 2)).round(2)
    else:
        df['bmi'] = 22.5

    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 7. Physiological Outlier Filtering (Remove Sensor Glitches)
    # Exclude biologically impossible or severely corrupted readings
    clean_filter = pd.Series(True, index=df.index)

    if 'heart_rate' in df.columns:
        clean_filter &= (df['heart_rate'] >= 30) & (df['heart_rate'] <= 220)
    if 'spo2' in df.columns:
        clean_filter &= (df['spo2'] >= 60) & (df['spo2'] <= 100)
    if 'temp_c' in df.columns:
        clean_filter &= (df['temp_c'] >= 32.0) & (df['temp_c'] <= 43.0)
    if 'age' in df.columns:
        clean_filter &= (df['age'] >= 1) & (df['age'] <= 110)

    df = df[clean_filter]
    print(f"[+] Rows retained after clinical outlier filtering: {len(df)}")

    # 8. Check Missing Values (Nulls)
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("\n[+] Imputing remaining null values using median strategies:")
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # 9. Clean Target / Risk Label Column
    target_candidates = ['risk_category', 'risk_level', 'status', 'target', 'condition', 'risk']
    found_target = None
    for cand in target_candidates:
        matched = [col for col in df.columns if cand == col or cand in col]
        if matched:
            found_target = matched[0]
            break

    if found_target:
        print(f"[+] Found target column: '{found_target}'")
        df.rename(columns={found_target: 'target_label'}, inplace=True)
        # If labels are strings, convert them to integer categories
        if not pd.api.types.is_numeric_dtype(df['target_label']):
            df['target_label'] = df['target_label'].astype(str).str.strip().str.title()
            categories = df['target_label'].unique()
            mapping = {cat: idx for idx, cat in enumerate(categories)}
            print(f"[+] Target label mapping: {mapping}")
            df['target_code'] = df['target_label'].map(mapping)
        else:
            df['target_code'] = df['target_label'].astype(int)
    else:
        print("[+] No existing label column found. Generating clinical condition labels...")
        conditions = [
            df.get('spo2', 98) < 92,
            df.get('temp_c', 36.8) > 38.3,
            (df.get('heart_rate', 75) > 105) | (df.get('heart_rate', 75) < 50)
        ]
        choices = [3, 1, 2]  # 3: Hypoxia, 1: Fever, 2: Tachy/Bradycardia
        df['target_code'] = np.select(conditions, choices, default=0)

    # 10. Save Cleaned Dataset
    df.to_csv(output_file, index=False)
    print(f"\n[✓] Cleaning complete! Cleaned dataset saved to: {output_file}")
    print(f"[✓] Final Cleaned Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\n--- Cleaned Data Sample (First 3 Rows) ---")
    print(df.head(3))

if __name__ == "__main__":
    clean_wban_dataset()