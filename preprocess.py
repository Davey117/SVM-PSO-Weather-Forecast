
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_clean_data(filepath="weatherAUS.csv"):
    """
    Ingests raw meteorological logs, enforces data integrity by scrubbing
    null targets, drops high-sparsity dimensions, and returns cleaned structures.
    """
    print(f"[*] Ingesting raw weather matrix from: {filepath}...")
    df = pd.read_csv(filepath)
    
    # 1. Enforce Ground Truth Integrity: Purge records missing the response label
    initial_rows = df.shape[0]
    df = df.dropna(subset=['RainTomorrow'])
    purged_rows = initial_rows - df.shape[0]
    if purged_rows > 0:
        print(f"[!] Purged {purged_rows} records due to missing 'RainTomorrow' target labels.")
        
    # 2. Structural Column Pruning: Eliminate dimensions with catastrophic missing percentages (>35%)
    # This prevents artificial imputation noise from blinding the subsequent SVM engine
    high_null_cols = ['Sunshine', 'Evaporation', 'Cloud9am', 'Cloud3pm', 'Date']
    print(f"[*] Dropping unviable/high-null dimensions: {high_null_cols}")
    df = df.drop(columns=high_null_cols, errors='ignore')
    
    return df


def execute_imputation_and_encoding(df):
    """
    Handles internal column modifications. Applies statistical imputation 
    to numeric vectors and encodes spatial/directional categorical arrays.
    """
    print("[*] Commencing statistical refinement and categorical serialization...")
    
    # Separate columns by structural behavior
    target_col = 'RainTomorrow'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_col]
    
    # Robust Median Imputation for Numeric Vectors to handle atmospheric skewed outliers
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            
    # Category Serialization: Handle spatial attributes and wind direction vectors
    # Fill categorical gaps with the mode (most frequent occurrence)
    label_encoders = {}
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            
        # Transform strings into distinct numerical integers for binary swarm processing
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    # Binary map the response target vector: Yes -> 1, No -> 0
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0})
    
    return df, numeric_cols, categorical_cols


def prepare_experimental_matrices(df, target_col='RainTomorrow', test_size=0.3, random_state=42):
    """
    Separates predictors from targets, scales the geometric feature space via
    StandardScaler, and returns stratified train/test partitions.
    """
    print("[*] Normalizing geometric features and partitioning workspace matrices...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col].values
    feature_names = X.columns.tolist()
    
    # Geometry Stabilization: Enforce zero-mean and unit variance scale across features.
    # This ensures that deep metrics like Pressure do not drown out shallow indices like Rainfall.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Stratified Data Split: Preserves class balance proportions across baseline matrices
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=test_size, 
        stratify=y, 
        random_state=random_state
    )
    
    print(f"[+] Matrix Partition Completed Successfully:")
    print(f"    -> Training Partition Shape : {X_train.shape}")
    print(f"    -> Testing Partition Shape  : {X_test.shape}")
    print(f"    -> Total Extracted Features  : {len(feature_names)} features")
    
    return X_train, X_test, y_train, y_test, feature_names


def get_preprocessed_pipeline(filepath="weatherAUS.csv"):
    """
    Unified execution pipeline to be called natively by your BPSO core and Streamlit frontend.
    """
    cleaned_df = load_and_clean_data(filepath)
    refined_df, _, _ = execute_imputation_and_encoding(cleaned_df)
    return prepare_experimental_matrices(refined_df)


if __name__ == "__main__":
    # Internal Unit Test for validation on your Linux workspace
    print("[*] Running standalone pipeline sanity check...")
    try:
        X_tr, X_te, y_tr, y_te, feats = get_preprocessed_pipeline("weatherAUS.csv")
        print("[+] Preprocessing pipeline sanity check passed flawlessly!")
    except Exception as e:
        print(f"[-] Defect identified during pipeline execution: {str(e)}")
