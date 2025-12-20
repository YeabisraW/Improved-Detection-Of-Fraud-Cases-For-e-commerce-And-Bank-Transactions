# src/utils.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ip_to_int(ip):
    try:
        return sum(int(x) * 256**i for i, x in enumerate(reversed(ip.split('.'))))
    except:
        return 0

def find_country(ip, ip_country_df=None):
    if ip == 0:
        return 'Unknown'
    if ip_country_df is None:
        raise ValueError("ip_country_df must be provided")
    row = ip_country_df[(ip_country_df['lower'] <= ip) & (ip_country_df['upper'] >= ip)]
    return row['country'].values[0] if not row.empty else 'Unknown'

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include='number')
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def plot_bivariate_categorical(df, cat_col, target_col='class'):
    plt.figure(figsize=(8,5))
    cross_tab = pd.crosstab(df[cat_col], df[target_col], normalize='index')
    cross_tab.plot(kind='bar', stacked=True)
    plt.title(f"{cat_col} vs {target_col}")
    plt.ylabel("Proportion")
    plt.show()

