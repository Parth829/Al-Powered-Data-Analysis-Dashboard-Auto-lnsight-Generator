import pandas as pd
import numpy as np

def clean_data(df):
    """
    Performs basic preprocessing on the dataframe:
    - Handles missing values
    - Removes duplicates
    - Converts obvious date columns
    
    Returns:
        tuple: (cleaned_dataframe, summary_string)
    """
    df_clean = df.copy()
    initial_rows, initial_cols = df_clean.shape
    
    # 1. Remove duplicates
    duplicates_count = df_clean.duplicated().sum()
    if duplicates_count > 0:
        df_clean = df_clean.drop_duplicates()
        
    # 2. Handle missing values
    missing_count = df_clean.isna().sum().sum()
    if missing_count > 0:
        # Numeric columns: fill with median (robust to outliers)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isna().sum() > 0:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
        # Categorical columns: fill with mode
        cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if df_clean[col].isna().sum() > 0:
                mode_val = df_clean[col].mode()
                if not mode_val.empty:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna("Unknown")
                    
    # 3. Detect and convert date columns
    date_converted_count = 0
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        # Simple heuristic: Check if column name suggests date or if values look like dates
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
                date_converted_count += 1
            except (ValueError, TypeError):
                pass
        else:
            # Try to infer if it's a date by testing first valid value
            first_valid = df_clean[col].dropna().iloc[0] if not df_clean[col].dropna().empty else None
            if isinstance(first_valid, str):
                # Basic check for typical date separators to avoid converting random strings
                if '-' in first_valid or '/' in first_valid:
                    try:
                        df_clean[col] = pd.to_datetime(df_clean[col])
                        date_converted_count += 1
                    except (ValueError, TypeError):
                        pass

    # Build summary string
    summary = []
    if duplicates_count > 0:
        summary.append(f"Removed {duplicates_count} duplicate rows.")
    if missing_count > 0:
        summary.append(f"Imputed {missing_count} missing values.")
    if date_converted_count > 0:
        summary.append(f"Converted {date_converted_count} column(s) to DateTime format.")
        
    if not summary:
        summary_str = "No major cleaning needed. Data looks good!"
    else:
        summary_str = " ".join(summary)
        
    return df_clean, summary_str
