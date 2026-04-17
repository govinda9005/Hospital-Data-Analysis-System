import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import timedelta

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)
Faker.seed(42)

fake = Faker()

def format_date(dt):
    # 3 different date formats mixed in date columns
    formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%b-%y']
    return dt.strftime(random.choice(formats))

# ----------------- Generate Patients Data -----------------
num_patients = 500
patient_ids = [f'P{i:03d}' for i in range(1, num_patients + 1)]
diagnoses = ['Hypertension', 'Diabetes', 'Pneumonia', 'Fracture', 'Appendicitis', 
             'Asthma', 'COVID-19', 'Heart Failure', 'Stroke', 'Migraine']

patients_data = []
for pid in patient_ids:
    admission_date = fake.date_time_between(start_date='-2y', end_date='now')
    stay_duration = timedelta(days=random.randint(1, 30))
    discharge_date = admission_date + stay_duration
    
    patients_data.append({
        'patient_id': pid,
        'name': fake.name(),
        'age': np.random.randint(18, 90),
        'gender': np.random.choice(['Male', 'Female']),
        'diagnosis': np.random.choice(diagnoses),
        'admission_date': format_date(admission_date),
        'discharge_date': format_date(discharge_date),
        'ward': np.random.choice(['General', 'ICU', 'Emergency', 'Surgery', 'Maternity'])
    })

patients_df = pd.DataFrame(patients_data)

# ----------------- Generate Billing Data -----------------
num_bills = 500
bill_ids = [f'B{i:03d}' for i in range(1, num_bills + 1)]

# 20 patient_ids in billing.csv that don't exist in patients.csv
valid_patient_ids = patient_ids[:-20]  # Take 480 valid IDs
invalid_patient_ids = [f'P{i:03d}' for i in range(1001, 1021)] # 20 non-existent ones
billing_patient_ids = random.choices(valid_patient_ids, k=480) + invalid_patient_ids
random.shuffle(billing_patient_ids) # keep it randomized

billing_data = []
for i, bid in enumerate(bill_ids):
    treatment_cost = round(np.random.uniform(100, 5000), 2)
    medication_cost = round(np.random.uniform(50, 2000), 2)
    total_amount = treatment_cost + medication_cost
    
    # Payment status: "Paid", "Pending", "Insurance Claimed" — weighted 50/30/20
    payment_status = np.random.choice(['Paid', 'Pending', 'Insurance Claimed'], p=[0.5, 0.3, 0.2])
    
    billing_data.append({
        'bill_id': bid,
        'patient_id': billing_patient_ids[i],
        'treatment_cost': treatment_cost,
        'medication_cost': medication_cost,
        'total_amount': total_amount,
        'payment_status': payment_status,
        'insurance_provider': np.random.choice(['Medicare', 'BlueCross', 'Aetna', 'Cigna', 'UnitedHealth', 'None'])
    })

billing_df = pd.DataFrame(billing_data)

# ----------------- Intentionally Include Issues -----------------

# 1. 15 outlier billing amounts (10x normal)
outlier_indices = random.sample(range(num_bills), 15)
for idx in outlier_indices:
    billing_df.at[idx, 'treatment_cost'] *= 10
    billing_df.at[idx, 'medication_cost'] *= 10
    billing_df.at[idx, 'total_amount'] *= 10

# 2. 40 missing values spread across columns
# Let's put 20 missing values in patients and 20 in billing
for _ in range(20):
    row_idx = random.randint(0, num_patients - 1)
    col_idx = random.choice(patients_df.columns[1:]) # exclude patient_id
    # Note: Setting to None generates NaN/NaT in pandas.
    patients_df.iat[row_idx, patients_df.columns.get_loc(col_idx)] = None

for _ in range(20):
    row_idx = random.randint(0, num_bills - 1)
    col_idx = random.choice(billing_df.columns[1:]) # exclude bill_id
    billing_df.iat[row_idx, billing_df.columns.get_loc(col_idx)] = None

# 3. 30 duplicate rows total (15 duplicates in patients, 15 duplicates in billing)
# We overwrite 15 randomly chosen rows with another 15 randomly chosen ones.
# Total row count remains 500.
patients_dup_targets = random.sample(range(num_patients), 15)
patients_dup_sources = random.sample([x for x in range(num_patients) if x not in patients_dup_targets], 15)
for t, s in zip(patients_dup_targets, patients_dup_sources):
    patients_df.iloc[t] = patients_df.iloc[s]

billing_dup_targets = random.sample(range(num_bills), 15)
billing_dup_sources = random.sample([x for x in range(num_bills) if x not in billing_dup_targets], 15)
for t, s in zip(billing_dup_targets, billing_dup_sources):
    billing_df.iloc[t] = billing_df.iloc[s]

# Save to /data folder
os.makedirs('data', exist_ok=True)
patients_df.to_csv('data/patients.csv', index=False)
billing_df.to_csv('data/billing.csv', index=False)

# ----------------- Print Summary -----------------
print("======= Data Generation Summary =======")
print(f"Patients data saved to data/patients.csv: {len(patients_df)} rows")
print(f"Billing data saved to data/billing.csv: {len(billing_df)} rows")
print("")
print("--- Null Counts per Column ---")
print("Patients:")
print(patients_df.isnull().sum())
print("\nBilling:")
print(billing_df.isnull().sum())
print("")
print("--- Date Format Variety Used in Patients ---")
print("Formats dynamically used: YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YY")
print("Sample rows showing dates:")
print(patients_df[['admission_date', 'discharge_date']].dropna().head(5).to_string(index=False))
print("=======================================")
