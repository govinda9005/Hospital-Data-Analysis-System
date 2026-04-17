import os
import logging
import pandas as pd
from sqlalchemy import Engine, text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _execute_query(engine: Engine, query: str, query_name: str) -> pd.DataFrame:
    """Helper method to run query via engine, read to DataFrame, and log."""
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        logger.info(f"Query '{query_name}' successfully executed. Rows returned: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Error executing query '{query_name}': {e}")
        return pd.DataFrame()

def avg_cost_by_diagnosis(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose: 
    Identifies the average financial cost (treatment and overall) associated with various 
    medical diagnoses. Helps hospital administration allocate budgets per department and 
    understand which medical conditions bear the heaviest financial weights.
    """
    query = """
        SELECT p.diagnosis, 
               ROUND(AVG(b.treatment_cost), 2) AS avg_treatment_cost, 
               ROUND(AVG(b.total_amount), 2) AS avg_total_amount
        FROM patients p
        JOIN billing b ON p.patient_id = b.patient_id
        GROUP BY p.diagnosis
        ORDER BY avg_total_amount DESC;
    """
    return _execute_query(engine, query, "Avg Cost by Diagnosis")

def monthly_admission_trends(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Analyzes historical admission volumes grouped by month. This trend analysis supports 
    seasonal capacity planning, specialized nurse scheduling, and resource stockpiling.
    """
    query = """
        SELECT DATE_TRUNC('month', admission_date) AS admission_month,
               COUNT(*) AS admissions_count
        FROM patients
        GROUP BY admission_month
        ORDER BY admission_month ASC;
    """
    return _execute_query(engine, query, "Monthly Admission Trends")

def top_diagnoses(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Highlights the Top 5 most frequent medical cases admitted to the hospital, including their 
    percentage share against total admissions. Vital for prioritizing clinical staffing 
    and specialized medical equipment readiness.
    """
    query = """
        WITH total_counts AS (
            SELECT COUNT(*) AS total FROM patients
        )
        SELECT diagnosis, 
               COUNT(*) AS diagnosis_count,
               ROUND((COUNT(*) * 100.0 / (SELECT total FROM total_counts)), 2) AS percentage
        FROM patients
        GROUP BY diagnosis
        ORDER BY diagnosis_count DESC
        LIMIT 5;
    """
    return _execute_query(engine, query, "Top Diagnoses")

def payment_status_breakdown(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Evaluates the hospital's revenue cycle health by quantifying the breakdown of invoice statuses. 
    A high volume of "Pending" might signal process bottlenecks in the billing department.
    """
    query = """
        WITH total_bills AS (
            SELECT COUNT(*) AS total FROM billing
        )
        SELECT payment_status, 
               COUNT(*) AS status_count,
               ROUND((COUNT(*) * 100.0 / (SELECT total FROM total_bills)), 2) AS percentage
        FROM billing
        GROUP BY payment_status
        ORDER BY status_count DESC;
    """
    return _execute_query(engine, query, "Payment Status Breakdown")

def avg_stay_duration_by_ward(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Measures the efficiency of patient turnover across different wards. Wards with excessively 
    high average stay durations can be inspected for discharge delays or intensive care necessities.
    """
    query = """
        SELECT ward, 
               ROUND(AVG(discharge_date - admission_date), 2) AS avg_stay_days
        FROM patients
        WHERE admission_date IS NOT NULL AND discharge_date IS NOT NULL
        GROUP BY ward
        ORDER BY avg_stay_days DESC;
    """
    return _execute_query(engine, query, "Avg Stay Duration by Ward")

def high_risk_patients(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Flags high-risk elderly patients (Age > 60) who have outstanding/pending payments. 
    This list is crucial for patient relations and financial counseling representatives to 
    provide targeted, compassionate assistance with medical debts.
    """
    query = """
        SELECT p.patient_id, p.name, p.age, p.diagnosis, b.total_amount
        FROM patients p
        JOIN billing b ON p.patient_id = b.patient_id
        WHERE p.age > 60 AND b.payment_status = 'Pending'
        ORDER BY p.age DESC;
    """
    return _execute_query(engine, query, "High Risk Patients")

def revenue_by_insurance_provider(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Quantifies the actual revenue successfully generated through insurance clams per provider 
    (where payment_status is 'Insurance Claimed'). Aids contract negotiations with insurance agencies.
    """
    query = """
        SELECT insurance_provider, 
               SUM(total_amount) AS total_revenue
        FROM billing
        WHERE payment_status = 'Insurance Claimed'
        GROUP BY insurance_provider
        ORDER BY total_revenue DESC;
    """
    return _execute_query(engine, query, "Revenue by Insurance Provider")

def patients_without_billing(engine: Engine) -> pd.DataFrame:
    """
    Business Purpose:
    Performs an internal integrity check to find patients who were admitted but have no corresponding 
    billing profile. Acts as an operational safety net for the auditing team to prevent unbilled treatments.
    """
    query = """
        SELECT p.patient_id, p.name, p.admission_date, p.ward
        FROM patients p
        LEFT JOIN billing b ON p.patient_id = b.patient_id
        WHERE b.bill_id IS NULL;
    """
    return _execute_query(engine, query, "Patients Without Billing")


def run_all_queries(engine: Engine) -> None:
    """
    Executes all 8 analytical queries, prints formatted results, and saves DataFrames securely 
    to the /reports/query_results/ directory as CSVs.
    """
    logger.info("=== Starting Analytical Queries Pipeline ===")
    
    reports_dir = "reports/query_results"
    os.makedirs(reports_dir, exist_ok=True)
    
    # Mapping of filenames to their respective analytical functions
    query_funcs = {
        "avg_cost_by_diagnosis": avg_cost_by_diagnosis,
        "monthly_admission_trends": monthly_admission_trends,
        "top_diagnoses": top_diagnoses,
        "payment_status_breakdown": payment_status_breakdown,
        "avg_stay_duration_by_ward": avg_stay_duration_by_ward,
        "high_risk_patients": high_risk_patients,
        "revenue_by_insurance_provider": revenue_by_insurance_provider,
        "patients_without_billing": patients_without_billing
    }
    
    for filename, func in query_funcs.items():
        logger.info(f"--- Running {filename} ---")
        df = func(engine)
        
        # Display nicely in console
        print(f"\n--- {filename.replace('_', ' ').title()} ---")
        if df.empty:
            print("No records returned.\n")
        else:
            print(df.to_string(index=False))
            print("\n")
            
        # Export as CSV
        out_path = os.path.join(reports_dir, f"{filename}.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved results to {out_path}")

    logger.info("=== Analytical Queries Pipeline Complete ===")

if __name__ == "__main__":
    from db_loader import create_engine_connection
    engine = create_engine_connection()
    if engine:
        run_all_queries(engine)
        engine.dispose()
