import os
import logging
import base64
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for server environments
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHARTS_DIR = 'reports/charts'
os.makedirs(CHARTS_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')

def _save_figure(path: str) -> None:
    """Helper method to apply tight_layout, save the figure at 150 DPI, and close it."""
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"Successfully saved chart to {path}")

def plot_top_diagnoses(df: pd.DataFrame) -> str:
    """
    Business Purpose: Transforms the top diagnoses subset into an intuitive horizontal bar chart.
    Highlights frequency volumes distinctly with a teal palette to help medical planners.
    
    Returns:
        str: File path to the saved chart.
    """
    logger.info("Plotting Top Diagnoses")
    
    # Needs a figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort ascending for horizontal bar chart so biggest is at top
    df_sorted = df.sort_values(by='diagnosis_count', ascending=True)
    
    bars = ax.barh(df_sorted['diagnosis'], df_sorted['diagnosis_count'], color='teal')
    
    # Add count labels on bars
    for bar in bars:
        ax.annotate(str(bar.get_width()), 
                    xy=(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2),
                    va='center', ha='left', fontsize=10)
        
    ax.set_title('Most common diagnoses', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Admitted Patients')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    out_path = os.path.join(CHARTS_DIR, 'top_diagnoses.png')
    _save_figure(out_path)
    return out_path

def plot_monthly_admissions(df: pd.DataFrame) -> str:
    """
    Business Purpose: Produces a time-series line chart tracking hospital admissions over time. 
    Includes a linear trendline to visualize growth or decline trajectories.
    
    Returns:
        str: File path to the saved chart.
    """
    logger.info("Plotting Monthly Admissions")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Convert dates to string (month/year) for formatting if they aren't already string
    if pd.api.types.is_datetime64_any_dtype(df['admission_month']):
        x_labels = df['admission_month'].dt.strftime('%b %Y')
    else:
        df['admission_month'] = pd.to_datetime(df['admission_month'])
        x_labels = df['admission_month'].dt.strftime('%b %Y')

    y_vals = df['admissions_count'].values
    x_indices = np.arange(len(x_labels))
    
    # Main line with markers
    ax.plot(x_indices, y_vals, marker='o', linewidth=2, color='#1f77b4', label='Admissions')
    
    # Trend line using polyfit (degree 1)
    if len(x_indices) > 1:
        z = np.polyfit(x_indices, y_vals, 1)
        p = np.poly1d(z)
        ax.plot(x_indices, p(x_indices), "r--", linewidth=1.5, alpha=0.7, label='Trend')
        
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_title('Hospital Admissions Over Time', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total Admissions')
    ax.legend()
    
    out_path = os.path.join(CHARTS_DIR, 'monthly_admissions.png')
    _save_figure(out_path)
    return out_path

def plot_payment_status(df: pd.DataFrame) -> str:
    """
    Business Purpose: Generates a pie chart emphasizing the distribution of billing payment statuses.
    Explodes the thickest slice to draw the financial team's attention to primary outcome metrics.
    
    Returns:
        str: File path to the saved chart.
    """
    logger.info("Plotting Payment Status Breakdown")
    fig, ax = plt.subplots(figsize=(7, 7))
    
    sizes = df['status_count'].values
    labels = df['payment_status'].values
    
    # Explode the largest slice slightly
    max_idx = np.argmax(sizes)
    explode = [0.1 if i == max_idx else 0 for i in range(len(sizes))]
    
    colors = sns.color_palette("pastel")[0:len(sizes)]
    
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', 
           startangle=140, shadow=True, colors=colors)
    ax.set_title('Billing Payment Status Distribution', fontsize=14, fontweight='bold')
    
    out_path = os.path.join(CHARTS_DIR, 'payment_status.png')
    _save_figure(out_path)
    return out_path

def plot_cost_correlation_heatmap(patients_df: pd.DataFrame, billing_df: pd.DataFrame) -> str:
    """
    Business Purpose: Merges patients and billing datasets and produces a seaborn heatmap.
    Helps analytical teams immediately identify strong associations (e.g. does older age correlate 
    with dramatically higher treatment costs or longer stay durations?).
    
    Returns:
        str: File path to the saved chart.
    """
    logger.info("Plotting Cost Correlation Heatmap")
    
    # Merge datasets
    merged_df = pd.merge(patients_df, billing_df, on='patient_id', how='inner')
    
    # Calculate stay duration
    merged_df['admission_date'] = pd.to_datetime(merged_df['admission_date'])
    merged_df['discharge_date'] = pd.to_datetime(merged_df['discharge_date'])
    merged_df['stay_duration'] = (merged_df['discharge_date'] - merged_df['admission_date']).dt.days
    
    # Select columns for correlation
    cols_to_correlate = ['age', 'treatment_cost', 'medication_cost', 'total_amount', 'stay_duration']
    corr_matrix = merged_df[cols_to_correlate].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, square=True, ax=ax,
                linewidths=0.5, cbar_kws={"shrink": .8})
    
    ax.set_title('Correlation Heatmap Metrics', fontsize=14, fontweight='bold')
    
    out_path = os.path.join(CHARTS_DIR, 'correlation_heatmap.png')
    _save_figure(out_path)
    return out_path

def _image_to_base64(img_path: str) -> str:
    """Helper to convert chart images to base64 for standalone HTML embedding."""
    try:
        with open(img_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to encode image {img_path}: {e}")
        return ""

def generate_html_report(query_results_dict: Dict[str, pd.DataFrame], chart_paths: List[str]) -> str:
    """
    Business Purpose: Synthesizes all charts and query results into a beautiful, self-contained, 
    deployable HTML page. No external dependencies required to view it.
    
    Returns:
        str: File path to the generated HTML report.
    """
    logger.info("Generating standalone HTML Report")
    
    date_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    
    # CSS Styling
    css = """
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; background: #fff; padding: 30px; box-shadow: 0px 4px 12px rgba(0,0,0,0.1); border-radius: 8px; }
        h1 { color: #2c3e50; text-align: center; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 40px; font-style: italic; }
        h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; margin-top: 40px; }
        .grid { display: flex; flex-wrap: wrap; gap: 30px; margin-bottom: 40px; justify-content: center; }
        .chart-box { background: #fafafa; border: 1px solid #eee; padding: 10px; border-radius: 8px; box-shadow: inset 0px 0px 5px rgba(0,0,0,0.02); text-align: center; }
        .chart-box img { max-width: 100%; height: auto; border-radius: 4px; }
        table { width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 30px; font-size: 0.9em; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        th { background-color: #34495e; color: white; padding: 12px 15px; text-align: left; }
        td { padding: 12px 15px; border-bottom: 1px solid #ecf0f1; }
        tr:hover { background-color: #f5f6fa; }
        .footer { text-align: center; margin-top: 50px; color: #95a5a6; font-size: 0.85em; }
    </style>
    """
    
    # Build HTML Content
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hospital Data Analysis Report</title>
    {css}
</head>
<body>
    <div class="container">
        <h1>🏥 Comprehensive Hospital Data Analysis</h1>
        <div class="subtitle">Generated dynamically on: {date_str}</div>

        <h2>Executive Visual Insights</h2>
        <div class="grid">
    """
    
    # Embed Images
    for path in chart_paths:
        file_name = os.path.basename(path).replace(".png", "").replace("_", " ").title()
        b64_str = _image_to_base64(path)
        if b64_str:
            html += f"""
            <div class="chart-box">
                <img src="data:image/png;base64,{b64_str}" alt="{file_name}">
            </div>
            """
            
    html += """
        </div>
        <h2>Analytical Query Outcomes</h2>
    """
    
    # Embed DataFrames as HTML tables
    for title, df in query_results_dict.items():
        html += f"<h3>{title}</h3>\n"
        if df.empty:
            html += "<p><em>No records found for this query.</em></p>\n"
        else:
            # Drop index and format as HTML
            table_html = df.to_html(index=False, classes='table')
            # Simply strip the pandas border mapping to ensure our CSS takes point
            table_html = table_html.replace('border="1"', '')
            html += table_html
            
    html += """
        <div class="footer">
            Confidential Hospital Report System | 2026
        </div>
    </div>
</body>
</html>
    """
    
    report_path = 'reports/hospital_analysis_report.html'
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"Successfully generated full standalone HTML report at: {report_path}")
    except Exception as e:
        logger.error(f"Error saving HTML report: {e}")
        
    return report_path

if __name__ == "__main__":
    # Provides an ad-hoc executable environment simply demonstrating structure.
    # In full system integration, this would consume the 'engine' and dfs natively.
    pass
