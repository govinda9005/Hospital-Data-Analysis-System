# Hospital Data Analysis System 🏥

> An end-to-end Python operational pipeline for processing, sanitizing, and structurally analyzing hospital admission flows and billing ledgers.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) 
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

## 💻 Tech Stack
- **Data Processing:** Python, Pandas, NumPy
- **Database:** PostgreSQL, SQLAlchemy (Engine)
- **Data Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, dotenv

## 🔑 Key Features
* **Automated Data Sanitization:** Algorithmic isolation of missing constraints (via median/flag imputation), 10x-outlier IQR mathematical capping, and explicit referential-integrity mapping.
* **SQLAlchemy Database Architecture:** Secure environment integration leveraging PostgreSQL to build relational tables cleanly.
* **Analytical Matrix Computations:** Eight precise, macro-level analytical SQL inquiries targeting stay-duration modeling, billing bottlenecks, and vulnerable demographics isolation.
* **Publication-Quality Insights:** Configured multi-vector Seaborn & Matplotlib frameworks constructing dynamic visualization models directly to embedded standalone HTML files.
* **Jupyter Narrative Workflow:** A highly structured, explainer-driven notebook engineered to visually decode operational inefficiencies for executives.

## 🏗️ Architecture Highlights
- **End-to-End Pipeline:** Full python automated ingestion, sanitization, and database integration loop.
- **Relational Database Design:** Proper PostgreSQL schema definition utilizing SQLAlchemy.
- **Cascade Deletes:** Foreign Keys natively configured for patient deletion consistency.
- **Multi-step Sanitization:** Algorithmic isolation of missing constraints and mathematical outlier removal.
- **Analytical Reporting:** Production-ready HTML standalone output dynamically generated.

## 🗄️ Database Design

### `patients`
- `patient_id` (Primary Key)
- `name`
- `age`
- `gender`
- `diagnosis`
- `admission_date`
- `discharge_date`
- `ward`

### `billing`
- `bill_id` (Primary Key)
- `patient_id` (Foreign Key)
- `treatment_cost`
- `medication_cost`
- `total_amount`
- `payment_status`
- `insurance_provider`

**Relationship:**
- One Patient → Many Billings (Cascade delete enabled)

## 📂 Project Structure

```text
hospital-data-analysis/
├── data/
│   ├── patients.csv               # Raw generated mock demographics
│   ├── billing.csv                # Raw generated mock financials
│   └── cleaned/                   # Sanitized operational outputs
├── scripts/
│   ├── data_generator.py          # Synthetic clinical ingestion engine
│   ├── data_cleaning.py           # IQR/Missingness normalization pipelines
│   ├── db_loader.py               # PostgreSQL schema & ETL framework
│   ├── queries.py                 # Core business query analytical suites
│   └── insights.py                # Visual modeling & HTML report genesis
├── notebooks/
│   └── analysis.ipynb             # Master analytical narrative
├── reports/
│   ├── charts/                    # Standalone .png render outputs
│   └── hospital_analysis_report.html # Integrated analytical executive brief
├── .env.example                   # PostgreSQL Template Variables
├── .gitignore                     # Git tracking exclusions
├── requirements.txt               # Pinned Python package environment
└── README.md
```

## 🚀 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/hospital-data-analysis.git
   cd hospital-data-analysis
   ```

2. **Initialize Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Windows: venv\Scripts\activate)
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the Database Environment**
   * Copy the `.env.example` configurations into a local system `.env` file. (Note: `.env` is already configured in `.gitignore` to prevent accidental credential leaking on GitHub).
   * Modify variables (`DB_USER`, `DB_PASSWORD`) matching your local PostgreSQL authentication.

## ⚙️ Usage Workflow

Execute the analytical pipeline linearly to ensure relational preservation:

1. **Synthetic Data Load:** Create the fundamental raw datasets.
   ```bash
   python scripts/data_generator.py
   ```
2. **Clinical Sanitization:** Cleanse formatting & mathematical outliers.
   ```bash
   python scripts/data_cleaning.py
   ```
3. **Database Architecture:** Deploy schema layouts and inject cleaned records.
   ```bash
   python scripts/db_loader.py
   ```
4. **Analytical Processing:** Extract operational vectors (exports natively to CSV).
   ```bash
   python scripts/queries.py
   ```
5. **Visualization Generation:** Render high-definition metrics charts and HTML reporting.
   ```bash
   python scripts/insights.py
   ```
6. **Executive Summary Review:** Walkthrough the structural interpretation.
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

## 📊 Sample Outputs

* **Cost Correlation Heatmaps:** Evaluates `patient_age` multi-vector mappings tightly against `stay_duration` proving a standardized +0.54 tracking reliance mapped directly to critical ICU strain markers natively exposed via `<coolwarm>` thresholds.
* **Payment Status Distribution:** A dynamic pie chart dissecting our accounts receivable pipeline, exposing that nearly 30% of accounts freeze within the `"Pending"` boundary, triggering internal administrative friction audits.

## ⚠️ Disclaimer
> **Data Privacy:** The clinical datasets (`patients.csv`, `billing.csv`) generated within this repository are 100% synthetically fabricated using the Faker library. No real Protected Health Information (PHI) or HIPAA-sensitive data is processed, utilized, or stored.

## 💼 Résumé Application
> Engineered a functional, end-to-end Python/PostgreSQL operational data-analysis pipeline capturing mock clinical admission matrices; standardized missingness through IQR capping and structured Pandas normalization culminating inside an intuitive, publication-ready Jupyter analytics presentation resolving administrative billing bottlenecks.

## 📝 License
This project operates under the **MIT License**.

## 🧑‍💻 Author
**Govinda Yadav**
