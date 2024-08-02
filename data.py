import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import quantstats as qs
import base64
from io import BytesIO
from bs4 import BeautifulSoup
from datetime import datetime
import math


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
    three_y = (
        (returns[-756:] + 1).prod() ** (252 / 756) - 1
        if len(returns) >= 756
        else np.nan
    )
    five_y = (
        (returns[-1260:] + 1).prod() ** (252 / 1260) - 1
        if len(returns) >= 1260
        else np.nan
    )
    ten_y = (
        (returns[-2520:] + 1).prod() ** (252 / 2520) - 1
        if len(returns) >= 2520
        else np.nan
    )

    summary = {
        "CAGR": cagr,
        "Volatility (ann.)": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "MTD": mtd,
        "3M": three_m,
        "6M": six_m,
        "YTD": ytd,
        "1Y": one_y,
        "3Y (ann.)": three_y,
        "5Y (ann.)": five_y,
        "10Y (ann.)": ten_y,
        "All-time (ann.)": cagr,  # Assuming All-time is same as CAGR
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
        "CAGR": cagr,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
    }
    return detailed


# Function to convert plot to base64 string
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str


# Function to plot cumulative returns manually
def plot_combined_cumulative_returns(sheet_names, field_names, title):
    fig, ax = plt.subplots(figsize=(16, 8))

    for strategy, field in zip(sheet_names, field_names):
        data = pd.read_excel("input.xlsx", sheet_name=strategy)
        returns = pd.Series(data[field].to_numpy(), index=data["date"])
        print(type(returns), "#################")
        returns = returns.dropna()
        cumulative_returns = (returns + 1).cumprod() - 1
        ax.plot(
            cumulative_returns.index, cumulative_returns, label=strategy, linewidth=1
        )
    # for strategy, returns in strategies_data.items():
    #     returns = returns.dropna()
    #     cumulative_returns = (returns + 1).cumprod() - 1
    #     ax.plot(
    #         cumulative_returns.index, cumulative_returns, label=strategy, linewidth=1
    #     )

    start_date = "2015-12-01"
    end_date = "2016-03-31"
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    ax.axvspan(
        start_date,
        end_date,
        color="red",
        alpha=0.3,
        label=start_date + "--" + end_date,
    )

    start_date = "2020-02-01"
    end_date = "2020-03-31"
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    ax.axvspan(
        start_date,
        end_date,
        color="red",
        alpha=0.3,
        label=start_date + "--" + end_date,
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.grid(True)
    ax.legend(fontsize="large")
    plt.tight_layout()
    return plot_to_base64(fig)


# Function to plot drawdown manually
def plot_combined_drawdown(sheet_names, field_names, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    # KHnB
    for strategy, field in zip(sheet_names, field_names):
        data = pd.read_excel("input.xlsx", sheet_name=strategy)
        returns = pd.Series(data[field].to_numpy(), index=data["date"])

        drawdown = qs.stats.to_drawdown_series(returns)
        ax.plot(drawdown.index, drawdown, label=strategy, linewidth=1)

    start_date = "2017-12-01"
    end_date = "2019-03-31"
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    ax.axvspan(
        start_date,
        end_date,
        color="red",
        alpha=0.3,
        label=start_date + "--" + end_date,
    )

    start_date = "2020-02-01"
    end_date = "2020-03-31"
    d1 = datetime.strptime(start_date, "%Y-%m-%d")
    d2 = datetime.strptime(end_date, "%Y-%m-%d")
    ax.axvspan(
        start_date,
        end_date,
        color="red",
        alpha=0.3,
        label=start_date + "--" + end_date,
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    ax.legend(fontsize="large")
    # legend = ax.get_legend()
    # if legend:
    #     legend.remove()  # Remove legend textbox if it exists
    plt.tight_layout()
    return plot_to_base64(fig)


# Function to plot monthly returns heatmap manually
def plot_monthly_heatmap(returns, title):
    fig, ax = plt.subplots(figsize=(16, 4))
    monthly_returns = qs.stats.monthly_returns(returns)
    monthly_returns = monthly_returns.drop(columns=["EOY"])  # Remove EOY column
    global_vmax = monthly_returns.max().max()
    global_vmin = monthly_returns.min().min()

    sns.heatmap(
        monthly_returns,
        cmap="RdYlGn",
        vmin=global_vmin,
        vmax=global_vmax,
        center=(global_vmax + global_vmin) / 2,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 10},
        cbar=False,
    )
    ax.set_title(title)
    plt.tight_layout()
    return plot_to_base64(fig)


# Function to plot EOY returns
def plot_eoy_returns(returns, title):
    returns = returns.resample("Y").apply(lambda x: (x + 1).prod() - 1)
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.barplot(x=returns.index.year, y=returns.values, ax=ax, color="blue")
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Returns")
    ax.legend().remove()  # Remove seaborn legend if it exists
    ax.set_xticklabels(returns.index.year, rotation=0, ha="right")
    plt.tight_layout()
    return plot_to_base64(fig)


# Function to plot worst 5 drawdown periods
def plot_worst_5_drawdown_periods(returns, title):
    # returns = returns.drop(returns.index[1:2])
    fig, ax = plt.subplots(figsize=(16, 8))
    # drawdown_periods = (
    #     # qs.stats.drawdown_details(returns).sort_values(by='max drawdown', ascending=True)[:5]
    # )
    datas = qs.stats.to_drawdown_series(returns)
    drawdown_periods = (
      qs.stats.drawdown_details(datas).sort_values(by="max drawdown").head(5)
    )
    # drawdown_periods = (
    #     qs.stats.drawdown_details(returns).sort_values(by="max drawdown").head(5)
    # )
    # print(qs.stats.to_drawdown_series(returns).sort_values('max drawdown')[:5], '#')
    cumulative_returns = (returns + 1).cumprod() - 1
    ax.plot(
        cumulative_returns.index,
        cumulative_returns,
        linewidth=1,
        color="blue",
        label="Cumulative Returns",
    )

    for idx, row in drawdown_periods.iterrows():
        start_date = row["start"]
        end_date = row["end"]
        d1 = datetime.strptime(start_date, "%Y-%m-%d")
        d2 = datetime.strptime(end_date, "%Y-%m-%d")
        # return abs((d2 - d1).days)
        # print(abs((d2 - d1).days))
        ax.axvspan(
            start_date,
            end_date,
            color="red",
            alpha=0.3,
            label=f"Drawdown Period {abs((d2 - d1).days) + 1}",
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Returns")
    ax.grid(True)

    # Consolidate unique legends
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))  # Using dict to keep only unique labels
    ax.legend(
        unique_labels.values(), unique_labels.keys(), loc="best"
    )  # Update the legend

    plt.tight_layout()
    return plot_to_base64(fig)


# Function to plot underwater plot
def plot_underwater(returns, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    drawdown = qs.stats.to_drawdown_series(returns)
    ax.fill_between(drawdown.index, drawdown, color="red", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    plt.tight_layout()
    return plot_to_base64(fig)


# Function to plot daily active returns
def plot_daily_active_returns(returns, title):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(returns.index, returns, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Active Returns")
    ax.grid(True)
    plt.tight_layout()
    return plot_to_base64(fig)


# Function to extract detailed table from QuantStats HTML report
def extract_detailed_table(html_file):
    with open(html_file, "r") as file:
        soup = BeautifulSoup(file, "html.parser")
        tables = soup.find_all("table")
        detailed_table_html = ""
        for table in tables:
            detailed_table_html += str(table)
        return detailed_table_html

def table_calendar_year(returns, title):
  returns = returns.resample("Y").apply(lambda x: (x + 1).prod() - 1)
  print(returns.index.year)
  print(returns.values)
  
  years = returns.index.year
  returns = returns.values
  returns = [format_percent(r) for r in returns]

  # Combine years and returns into a structured format for the table
  data = np.array([returns])  # Create a 2D array where the returns are in a single row

  # Create a Matplotlib figure
  fig, ax = plt.subplots(figsize=(16, 1))  # Adjust the figure size as needed

  # Hide axes
  ax.axis('tight')
  ax.axis('off')

  # Create the table
  table = ax.table(cellText=data, colLabels=years, cellLoc='center', loc='center', )

  # Set the font size for the table
  table.auto_set_font_size(False)
  table.set_fontsize(16)

  # Optionally, set the table's cell colors or styles
  for (i, j), cell in table.get_celld().items():
      if j == 0:  # Only the first column (row labels)
          cell.set_facecolor('#ffffff')  # Light gray background for the row header
          cell.set_height(0.5)
      else:  # Data cells
          cell.set_facecolor('#ffffff')  # White background for data cells
          cell.set_height(0.5)
  # Adjust layout and show the table
  plt.tight_layout()
  return plot_to_base64(fig)

def get_excess_return_stats(file_path):
    xls = pd.ExcelFile(file_path)
    field_names = [
        "return_next_1m",
        "return_next_1m",
        "return_next_1m",
        "return_next_8m",
        "return_next_6m",
    ]
    sheet_names = xls.sheet_names
    excess_df = pd.DataFrame()
    for name, field in zip(sheet_names, field_names):
        returns = pd.read_excel(file_path, sheet_name=name)
        highvol_returns = returns[returns["episode"] == "highvol"]
        lowvol_returns = returns[returns["episode"] == "lowvol"]
        excess_stats = {
            "All_Count": math.floor(returns["date"].count()),
            "All_Avg return": format_percent(returns[field].mean()),
            "All_Standard Deviation": format_percent(returns[field].std()),
            "LowVol_Count": math.floor(lowvol_returns["date"].count()),
            "LowVol_Avg return": format_percent(lowvol_returns[field].mean()),
            "LowVol_Standard Deviation": format_percent(lowvol_returns[field].std()),
            "HighVol_Count": math.floor(highvol_returns["date"].count()),
            "HighVol_Avg return": format_percent(highvol_returns[field].mean()),
            "HighVol_Standard Deviation": format_percent(highvol_returns[field].std()),
        }
        excess_df[name] = pd.Series(excess_stats)
    return excess_df


# Sample Data for Multiple Strategies using QuantStats
tickers = ["AAPL", "MSFT", "GOOG"]
strategies_data = {ticker: qs.utils.download_returns(ticker) for ticker in tickers}

# Initialize DataFrames for combined statistics
summary_df = pd.DataFrame()

# Calculate statistics for each strategy and combine into DataFrames
for strategy, returns in strategies_data.items():
    returns.to_csv("init.txt", header=True)
    summary_stats = calculate_summary_stats(returns)
    summary_df[strategy] = pd.Series(summary_stats)

# Format the DataFrames
summary_df = summary_df.applymap(
    lambda x: format_percent(x) if isinstance(x, float) else x
)
# combined_df = combined_df.applymap(
#     lambda x:format_percent(x) if isinstance(x, float) else x
# )
# Convert DataFrames to HTML
html_summary_table = summary_df.to_html(classes="table table-striped", border=0)

# Excess Return Stats DataFrames by Episode
# combined_df = excess_return_df.to_html(classes="table table-striped, border=0")

# Custom HTML template
file_path = "input.xlsx"
excess_stats_table = get_excess_return_stats(file_path)
html_excess_stats_table = excess_stats_table.to_html(
    classes="table table-striped", border=0
)

field_names = ["return_1m", "return_1m", "return_1m", "return_8m", "return_6m"]
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names
# sheet_names = ['strat1', 'strat2', 'strat3']

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
    <h1>Excess Return Stats</h1>
    {html_excess_stats_table}
    <h1>Charts</h1>
"""

# Create combined cumulative returns plot
html_template += f'<div class="chart"><h2>Combined Cumulative Returns</h2><img src="data:image/png;base64,{plot_combined_cumulative_returns(sheet_names, field_names, "Combined Cumulative Returns")}" alt="Combined Cumulative Returns"></div>'

# Create combined drawdown plot
html_template += f'<div class="chart"><h2>Combined Drawdown</h2><img src="data:image/png;base64,{plot_combined_drawdown(sheet_names, field_names, "Combined Drawdown")}" alt="Combined Drawdown"></div>'

# Create individual monthly returns heatmaps
for name, field in zip(sheet_names, field_names):
    returns = pd.read_excel("input.xlsx", sheet_name=name)
    data = pd.Series(returns[field].to_numpy(), index=returns["date"])
    html_template += f'<div class="chart"><h2>{name} Monthly Returns</h2><img src="data:image/png;base64,{plot_monthly_heatmap(data, name + " Monthly Returns")}" alt="{name} Monthly Returns"></div>'

# Create individual EOY returns plots
for name, field in zip(sheet_names, field_names):
    returns = pd.read_excel("input.xlsx", sheet_name=name)
    data = pd.Series(returns[field].to_numpy(), index=returns["date"])
    html_template += f'<div class="chart"><h2>{name} EOY Returns</h2><img src="data:image/png;base64,{plot_eoy_returns(data, name + " EOY Returns")}" alt="{name} EOY Returns"></div>'

# Create individual worst 5 drawdown periods plots
for name, field in zip(sheet_names, field_names):
    returns = pd.read_excel('input.xlsx', sheet_name=name)
    result = returns.groupby('date', as_index=False)[field].agg(lambda x: x.sum() / 2)

    data = pd.Series(result[field].to_numpy(), index=result['date'])
    html_template += f'<div class="chart"><h2>{name} Worst 5 Drawdown Periods</h2><img src="data:image/png;base64,{plot_worst_5_drawdown_periods(data, name + " Worst 5 Drawdown Periods")}" alt="{name} Worst 5 Drawdown Periods"></div>'

# Create individual underwater plots
for name, field in zip(sheet_names, field_names):
    returns = pd.read_excel('input.xlsx', sheet_name=name)
    result = returns.groupby('date', as_index=False)[field].agg(lambda x: x.sum() / 2)

    data = pd.Series(result[field].to_numpy(), index=result['date'])
    html_template += f'<div class="chart"><h2>{name} Underwater Plot</h2><img src="data:image/png;base64,{plot_underwater(data, name + " Underwater Plot")}" alt="{name} Underwater Plot"></div>'


# Create Calendar Year Returns plots
for name, field in zip(sheet_names, field_names):
    returns = pd.read_excel("input.xlsx", sheet_name=name)
    data = pd.Series(returns[field].to_numpy(), index=returns["date"])
    html_template += f'<div class="chart"><h2>{name} Calendar Year Returns</h2><img src="data:image/png;base64,{table_calendar_year(data, name + " Calendar Year Returns")}" alt="{name} Calendar Year Returns"></div>'

# Create Calendar Year Volatility plots
for name, field in zip(sheet_names, field_names):
    returns = pd.read_excel("input.xlsx", sheet_name=name)
    data = pd.Series(returns[field].to_numpy(), index=returns["date"])
    html_template += f'<div class="chart"><h2>{name} Calendar Year Returns</h2><img src="data:image/png;base64,{table_calendar_year_volatility(data, name + " Calendar Year Returns")}" alt="{name} Calendar Year Returns"></div>'

# # Create individual daily active returns plots
# for name, field in zip(sheet_names, field_names):
#     returns = pd.read_excel('input.xlsx', sheet_name=name)
#     data = pd.Series(returns[field].to_numpy(), index=returns['date'])
#     html_template += f'<div class="chart"><h2>{name} Daily Active Returns</h2><img src="data:image/png;base64,{plot_daily_active_returns(data, name + "Daily Active Returns")}" alt="{name} Daily Active Returns"></div>'

# Generate QuantStats HTML reports for each strategy and extract detailed tables
detailed_table_html = ""
for strategy, returns in strategies_data.items():
    returns.to_csv("test.txt", header=True)
    temp_report_file = f"temp_quantstats_report_{strategy}.html"
    qs.reports.html(returns, output=temp_report_file, title=f"{strategy} Report")
    detailed_table_html += f"<div style='width:33.3333%; float:left'><h1>{strategy} Detailed Statistics</h1>"
    print(temp_report_file, "####")
    detailed_table_html += extract_detailed_table(temp_report_file) + f"</div>"

# Add the extracted detailed tables to the HTML report
html_template += f"""
    <h1>Detailed Statistics</h1>
    <div>
    {detailed_table_html}
    </div>
</body>
</html>
"""

# Save the final report with the tables and charts
with open("combined_financial_performance_report.html", "w") as file:
    file.write(html_template)

# Convert HTML to PDF (uncomment if needed)
# pdfkit.from_file('combined_financial_performance_report.html', 'combined_financial_performance_report.pdf')

print(
    "Static HTML and PDF reports for all strategies have been generated successfully."
)