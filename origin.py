import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
import base64
from io import BytesIO
from bs4 import BeautifulSoup

# Function to format numbers and percentages
def format_percent(x):
    return "{:.2f}%".format(x * 100)

def format_number(x):
    return "{:.2f}".format(x)

# Function to calculate summary statistics manually
def calculate_summary_stats(returns):
    returns = returns.dropna()
    cagr = qs.stats.cagr(returns)
    volatility = qs.stats.volatility(returns)
    sharpe = qs.stats.sharpe(returns)
    max_drawdown = qs.stats.max_drawdown(returns)
    mtd = (returns[-21:] + 1).prod() - 1
    three_m = (returns[-63:] + 1).prod() - 1
    six_m = (returns[-126:] + 1).prod() - 1
    ytd = (returns[returns.index.year == returns.index[-1].year] + 1).prod() - 1
    one_y = (returns[-252:] + 1).prod() - 1
    three_y = (returns[-756:] + 1).prod() ** (252/756) - 1 if len(returns) >= 756 else np.nan
    five_y = (returns[-1260:] + 1).prod() ** (252/1260) - 1 if len(returns) >= 1260 else np.nan
    ten_y = (returns[-2520:] + 1).prod() ** (252/2520) - 1 if len(returns) >= 2520 else np.nan

    summary = {
        'CAGR': cagr,
        'Volatility (ann.)': volatility,
        'Sharpe': sharpe,
        'Max Drawdown': max_drawdown,
        'MTD': mtd,
        '3M': three_m,
        '6M': six_m,
        'YTD': ytd,
        '1Y': one_y,
        '3Y (ann.)': three_y,
        '5Y (ann.)': five_y,
        '10Y (ann.)': ten_y,
        'All-time (ann.)': cagr  # Assuming All-time is same as CAGR
    }
    return summary

# Function to calculate detailed statistics
def calculate_detailed_stats(returns):
    returns = returns.dropna()
    cagr = qs.stats.cagr(returns)
    volatility = qs.stats.volatility(returns)
    max_drawdown = qs.stats.max_drawdown(returns)
    sharpe = qs.stats.sharpe(returns)

    detailed = {
        'CAGR': cagr,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Sharpe Ratio': sharpe
    }
    return detailed

# Function to convert plot to base64 string
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# Function to plot cumulative returns manually
def plot_combined_cumulative_returns(strategies_data, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    for strategy, returns in strategies_data.items():
        returns = returns.dropna()
        cumulative_returns = (returns + 1).cumprod() - 1
        ax.plot(cumulative_returns.index, cumulative_returns, label=strategy, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.grid(True)
    ax.legend(fontsize='large')
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to plot drawdown manually
def plot_combined_drawdown(strategies_data, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    for strategy, returns in strategies_data.items():
        drawdown = qs.stats.to_drawdown_series(returns)
        ax.plot(drawdown.index, drawdown, label=strategy, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.grid(True)
    legend = ax.get_legend()
    if legend:
        legend.remove()  # Remove legend textbox if it exists
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to plot monthly returns heatmap manually
def plot_monthly_heatmap(returns, title):
    fig, ax = plt.subplots(figsize=(16, 12))
    monthly_returns = qs.stats.monthly_returns(returns)
    monthly_returns = monthly_returns.drop(columns=['EOY'])  # Remove EOY column
    sns.heatmap(monthly_returns, cmap='RdYlGn', center=0, annot=True, fmt=".2f", linewidths=0.5, ax=ax, annot_kws={"size": 10}, cbar=False)
    ax.set_title(title)
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to plot EOY returns
def plot_eoy_returns(returns, title):
    returns = returns.resample('Y').apply(lambda x: (x + 1).prod() - 1)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=returns.index.year, y=returns.values, ax=ax, color="blue")
    ax.set_title(title)
    ax.set_xlabel('Year')
    ax.set_ylabel('Returns')
    ax.legend().remove()  # Remove seaborn legend if it exists
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to plot worst 5 drawdown periods
def plot_worst_5_drawdown_periods(returns, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    drawdown_periods = qs.stats.drawdown_details(returns).sort_values(by='max drawdown').head(5)
    cumulative_returns = (returns + 1).cumprod() - 1
    ax.plot(cumulative_returns.index, cumulative_returns, linewidth=1, color='blue')

    for idx, row in drawdown_periods.iterrows():
        start_date = row['start']
        end_date = row['end']
        ax.axvspan(start_date, end_date, color='red', alpha=0.3, label=f'Period {idx+1}')
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.grid(True)
    legend = ax.get_legend()
    if legend:
        legend.remove()  # Remove legend textbox if it exists
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to plot underwater plot
def plot_underwater(returns, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    drawdown = qs.stats.to_drawdown_series(returns)
    ax.fill_between(drawdown.index, drawdown, color='red', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.grid(True)
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to plot daily active returns
def plot_daily_active_returns(returns, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(returns.index, returns, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Active Returns')
    ax.grid(True)
    plt.tight_layout()
    return plot_to_base64(fig)

# Function to extract detailed table from QuantStats HTML report
def extract_detailed_table(html_file):
    with open(html_file, 'r') as file:
        soup = BeautifulSoup(file, 'html.parser')
        tables = soup.find_all('table')
        detailed_table_html = ""
        for table in tables:
            detailed_table_html += str(table)
        return detailed_table_html

# Sample Data for Multiple Strategies using QuantStats
tickers = ['AAPL', 'MSFT', 'GOOG']
strategies_data = {ticker: qs.utils.download_returns(ticker) for ticker in tickers}

# Initialize DataFrames for combined statistics
summary_df = pd.DataFrame()

# Calculate statistics for each strategy and combine into DataFrames
for strategy, returns in strategies_data.items():
    summary_stats = calculate_summary_stats(returns)
    summary_df[strategy] = pd.Series(summary_stats)

# Format the DataFrames
summary_df = summary_df.applymap(lambda x: format_percent(x) if isinstance(x, float) else x)

# Convert DataFrames to HTML
html_summary_table = summary_df.to_html(classes="table table-striped", border=0)

# Custom HTML template
html_template = f"""
<html>
<head>
    <title>Combined Financial Performance Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .table th, .table td {{
            border: 1px solid #dddddd;
            text-align: right;
            padding: 8px;
        }}
        .table th {{
            background-color: #f2f2f2;
        }}
        .table td:first-child {{
            text-align: left;
        }}
        .table th:first-child {{
            text-align: left;
        }}
        .chart {{
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>Summary Statistics</h1>
    {html_summary_table}
    <h1>Charts</h1>
"""

# Create combined cumulative returns plot
html_template += f'<div class="chart"><h2>Combined Cumulative Returns</h2><img src="data:image/png;base64,{plot_combined_cumulative_returns(strategies_data, "Combined Cumulative Returns")}" alt="Combined Cumulative Returns"></div>'

# Create combined drawdown plot
html_template += f'<div class="chart"><h2>Combined Drawdown</h2><img src="data:image/png;base64,{plot_combined_drawdown(strategies_data, "Combined Drawdown")}" alt="Combined Drawdown"></div>'

# Create individual monthly returns heatmaps
for strategy, returns in strategies_data.items():
    html_template += f'<div class="chart"><h2>{strategy} Monthly Returns</h2><img src="data:image/png;base64,{plot_monthly_heatmap(returns, strategy + " Monthly Returns")}" alt="{strategy} Monthly Returns"></div>'

# Create individual EOY returns plots
for strategy, returns in strategies_data.items():
    html_template += f'<div class="chart"><h2>{strategy} EOY Returns</h2><img src="data:image/png;base64,{plot_eoy_returns(returns, strategy + " EOY Returns")}" alt="{strategy} EOY Returns"></div>'

# Create individual worst 5 drawdown periods plots
for strategy, returns in strategies_data.items():
    html_template += f'<div class="chart"><h2>{strategy} Worst 5 Drawdown Periods</h2><img src="data:image/png;base64,{plot_worst_5_drawdown_periods(returns, strategy + " Worst 5 Drawdown Periods")}" alt="{strategy} Worst 5 Drawdown Periods"></div>'

# Create individual underwater plots
for strategy, returns in strategies_data.items():
    html_template += f'<div class="chart"><h2>{strategy} Underwater Plot</h2><img src="data:image/png;base64,{plot_underwater(returns, strategy + " Underwater Plot")}" alt="{strategy} Underwater Plot"></div>'

# Create individual daily active returns plots
html_template += f'<div class="chart"><h2>MSFT Daily Active Returns</h2><img src="data:image/png;base64,{plot_daily_active_returns(strategies_data["MSFT"], "MSFT Daily Active Returns")}" alt="MSFT Daily Active Returns"></div>'

# Generate QuantStats HTML reports for each strategy and extract detailed tables
detailed_table_html = ""
for strategy, returns in strategies_data.items():
    temp_report_file = f'temp_quantstats_report_{strategy}.html'
    qs.reports.html(returns, output=temp_report_file, title=f"{strategy} Report")
    detailed_table_html += f"<h1>{strategy} Detailed Statistics</h1>"
    detailed_table_html += extract_detailed_table(temp_report_file)

# Add the extracted detailed tables to the HTML report
html_template += f"""
    <h1>Detailed Statistics</h1>
    {detailed_table_html}
</body>
</html>
"""

# Save the final report with the tables and charts
with open("combined_financial_performance_report.html", "w") as file:
    file.write(html_template)

# Convert HTML to PDF (uncomment if needed)
# pdfkit.from_file('combined_financial_performance_report.html', 'combined_financial_performance_report.pdf')

print("Static HTML and PDF reports for all strategies have been generated successfully.")
