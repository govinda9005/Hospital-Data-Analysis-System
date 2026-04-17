import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Tuple, Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(patients_path: str, billing_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load patients and billing datasets from the specified CSV file paths.

    Args:
        patients_path (str): Path to the patients CSV file.
        billing_path (str): Path to the billing CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the patients and billing DataFrames.
    """
    logger.info(f"Loading patients data from {patients_path}")
    patients_df = pd.read_csv(patients_path)
    
    logger.info(f"Loading billing data from {billing_path}")
    billing_df = pd.read_csv(billing_path)
    
    return patients_df, billing_df

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, int]: The cleaned DataFrame and the number of removed duplicate rows.
    """
    initial_len = len(df)
    cleaned_df = df.drop_duplicates().copy()
    count_removed = initial_len - len(cleaned_df)
    
    logger.info(f"Removed {count_removed} duplicate rows.")
    return cleaned_df, count_removed

def handle_missing_values(df: pd.DataFrame, strategy: str = 'flag') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fill missing values in the DataFrame. 
    Numeric columns are filled with the median.
    Categorical columns are filled with mode (if strategy='mode') or flagged as 'Unknown' (if strategy='flag').

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): Strategy for categorical columns ('flag' or 'mode'). Defaults to 'flag'.

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: The cleaned DataFrame and a summary dictionary of missing values filled.
    """
    cleaned_df = df.copy()
    summary_dict = {'numeric_filled_median': {}, 'categorical_filled': {}}
    
    for col in cleaned_df.columns:
        missing_count = cleaned_df[col].isnull().sum()
        if missing_count > 0:
            if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                median_val = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(median_val)
                summary_dict['numeric_filled_median'][col] = int(missing_count)
                logger.info(f"Filled {missing_count} missing values in numeric column '{col}' with median {median_val}.")
            else:
                if strategy == 'mode':
                    mode_val = cleaned_df[col].mode()[0]
                    cleaned_df[col] = cleaned_df[col].fillna(mode_val)
                    summary_dict['categorical_filled'][col] = int(missing_count)
                    logger.info(f"Filled {missing_count} missing values in categorical column '{col}' with mode '{mode_val}'.")
                else:
                    cleaned_df[col] = cleaned_df[col].fillna("Unknown")
                    summary_dict['categorical_filled'][col] = int(missing_count)
                    logger.info(f"Flagged {missing_count} missing values in categorical column '{col}' as 'Unknown'.")
                    
    return cleaned_df, summary_dict

def standardize_dates(df: pd.DataFrame, date_cols: List[str]) -> pd.DataFrame:
    """
    Convert specified date columns to datetime format and drop rows where all date columns are null.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_cols (List[str]): List of column names to standardize as dates.

    Returns:
        pd.DataFrame: The DataFrame with standardized dates.
    """
    cleaned_df = df.copy()
    initial_len = len(cleaned_df)
    
    for col in date_cols:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors='coerce')
            logger.info(f"Standardized date column '{col}'.")
            
    # Drop rows where BOTH/ALL specified date columns are null
    existing_date_cols = [c for c in date_cols if c in cleaned_df.columns]
    if existing_date_cols:
        cleaned_df = cleaned_df.dropna(subset=existing_date_cols, how='all')
    
    dropped_count = initial_len - len(cleaned_df)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows where all date columns {existing_date_cols} were null.")
        
    return cleaned_df

def remove_outliers_iqr(df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, int]:
    """
    Cap outliers in a numeric column using the Interquartile Range (IQR) method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The numeric column to cap outliers for.

    Returns:
        Tuple[pd.DataFrame, int]: The DataFrame with outliers capped and the count of capped outliers.
    """
    cleaned_df = df.copy()
    if col not in cleaned_df.columns or not pd.api.types.is_numeric_dtype(cleaned_df[col]):
        logger.warning(f"Column '{col}' is missing or not numeric. Skipping outlier treatment.")
        return cleaned_df, 0
        
    Q1 = cleaned_df[col].quantile(0.25)
    Q3 = cleaned_df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers_mask = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound))
    outlier_count = int(outliers_mask.sum())
    
    # Cap outliers
    cleaned_df.loc[cleaned_df[col] < lower_bound, col] = lower_bound
    cleaned_df.loc[cleaned_df[col] > upper_bound, col] = upper_bound
    
    logger.info(f"Capped {outlier_count} outliers in column '{col}' using bounds [{lower_bound:.2f}, {upper_bound:.2f}].")
    return cleaned_df, outlier_count

def validate_referential_integrity(patients_df: pd.DataFrame, billing_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Find billing records that have no matching patient_id in the patients DataFrame.

    Args:
        patients_df (pd.DataFrame): The patients DataFrame.
        billing_df (pd.DataFrame): The billing DataFrame.

    Returns:
        Dict[str, Any]: A report dictionary detailing the referential integrity check.
    """
    valid_patient_ids = set(patients_df['patient_id'].dropna())
    billing_patient_ids = set(billing_df['patient_id'].dropna())
    
    invalid_billing_records = billing_df[~billing_df['patient_id'].isin(valid_patient_ids)]
    invalid_count = len(invalid_billing_records)
    
    report = {
        'total_billing_records': len(billing_df),
        'records_without_valid_patient': invalid_count,
        'invalid_patient_ids_found': list(invalid_billing_records['patient_id'].unique())
    }
    
    if invalid_count > 0:
        logger.warning(f"Found {invalid_count} billing records with patient IDs not present in patients data.")
    else:
        logger.info("Referential integrity check passed: all billing patient IDs exist in patients data.")
        
    return report

def generate_cleaning_report(dataset_name: str, original_df: pd.DataFrame, cleaned_df: pd.DataFrame, actions_log: Dict[str, Any]) -> None:
    """
    Print and save a JSON summary of all changes made during the data cleaning process.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'patients', 'billing').
        original_df (pd.DataFrame): The DataFrame prior to cleaning.
        cleaned_df (pd.DataFrame): The DataFrame after cleaning.
        actions_log (Dict[str, Any]): Dictionary capturing all cleaning actions taken.
    """
    report = {
        'dataset': dataset_name,
        'original_rows': len(original_df),
        'cleaned_rows': len(cleaned_df),
        'removed_rows': len(original_df) - len(cleaned_df),
        'actions': actions_log
    }
    
    logger.info(f"\n=== Cleaning Report: {dataset_name.upper()} ===")
    logger.info("\n" + json.dumps(report, indent=4))
    
    report_path = f"data/cleaned/{dataset_name}_cleaning_report.json"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Cleaning report saved to {report_path}")

def clean_pipeline(patients_path: str, billing_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run all cleaning steps in order, save cleaned CSVs to /data/cleaned/, and return cleaned DataFrames.

    Args:
        patients_path (str): Path to the raw patients CSV file.
        billing_path (str): Path to the raw billing CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The fully cleaned patients and billing DataFrames.
    """
    logger.info("=== Starting Data Cleaning Pipeline ===")
    
    # 1. Load Data
    patients_df, billing_df = load_data(patients_path, billing_path)
    
    # Keeping copies for final reporting
    orig_patients = patients_df.copy()
    orig_billing = billing_df.copy()
    
    patients_actions = {}
    billing_actions = {}
    
    # --- CLEAN PATIENTS DATA ---
    logger.info("--- Cleaning Patients Data ---")
    patients_df, p_dups = remove_duplicates(patients_df)
    patients_actions['duplicates_removed'] = p_dups
    
    patients_df, p_missing = handle_missing_values(patients_df, strategy='flag')
    patients_actions['missing_values_handled'] = p_missing
    
    # Standardize dates
    patients_df = standardize_dates(patients_df, ['admission_date', 'discharge_date'])
    patients_actions['dates_standardized'] = ['admission_date', 'discharge_date']
    
    # --- CLEAN BILLING DATA ---
    logger.info("--- Cleaning Billing Data ---")
    billing_df, b_dups = remove_duplicates(billing_df)
    billing_actions['duplicates_removed'] = b_dups
    
    billing_df, b_missing = handle_missing_values(billing_df, strategy='flag')
    billing_actions['missing_values_handled'] = b_missing
    
    # Remove outliers for treatment_cost, medication_cost, and total_amount
    for col in ['treatment_cost', 'medication_cost', 'total_amount']:
        billing_df, out_count = remove_outliers_iqr(billing_df, col)
        billing_actions[f'outliers_capped_{col}'] = out_count
        
    # Standardize dates (if billing had any, but we know it doesn't currently, safe to leave empty or skip)
    # If billing had a date column we could add it here.
        
    # --- VALIDATE INTEGRITY ---
    logger.info("--- Validating Referential Integrity ---")
    integrity_report = validate_referential_integrity(patients_df, billing_df)
    billing_actions['referential_integrity'] = integrity_report
    
    # Save cleaned DataFrames
    os.makedirs('data/cleaned', exist_ok=True)
    
    patients_out_path = 'data/cleaned/patients_cleaned.csv'
    billing_out_path = 'data/cleaned/billing_cleaned.csv'
    
    patients_df.to_csv(patients_out_path, index=False)
    billing_df.to_csv(billing_out_path, index=False)
    logger.info(f"Cleaned datasets saved to {patients_out_path} and {billing_out_path}")
    
    # Generate reports
    generate_cleaning_report('patients', orig_patients, patients_df, patients_actions)
    generate_cleaning_report('billing', orig_billing, billing_df, billing_actions)
    
    logger.info("=== Data Cleaning Pipeline Complete ===")
    return patients_df, billing_df

if __name__ == '__main__':
    # Execute pipeline on the sample datasets generated in Session 1
    # Note: Requires the 'data' directory to have the initial raw csv files.
    if os.path.exists('data/patients.csv') and os.path.exists('data/billing.csv'):
        clean_pipeline('data/patients.csv', 'data/billing.csv')
    else:
        logger.error("Raw data files not found. Please run data_generator.py first.")
