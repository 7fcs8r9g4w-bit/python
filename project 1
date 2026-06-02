# -*- coding: utf-8 -*-
"""
Advanced Python Project: Carbon Emission Data Analysis
Prepared by: Mohammad Anwar Sayyah
Student ID: 202411349
Instructors: Dr. Ali Azwai & Dr. Ala Abuthawabeh
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Set up visual parameters
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def load_data(file_path):
    print("[1/7] Loading dataset...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Please check the path.")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def basic_statistics(df):
    print("\n[2/7] Computing Basic Statistics...")
    stat_cols = [
        'Total_Energy_Consumption_kWh', 
        'Renewable_Energy_Consumption_kWh', 
        'NonRenewable_Energy_Consumption_kWh', 
        'Production_Output_Units', 
        'Supply_Chain_Transport_km'
    ]
    
    stats = df[stat_cols].agg(['mean', 'median', 'max', 'min']).T.round(2)
    print("\n--- Statistical Metrics Summary ---")
    print(stats)
    return stat_cols

def plot_distributions(df, columns):
    print("\n[3/7] Generating Distribution Plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color='#2B6CB0')
        axes[i].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')
        
    # Remove the empty 6th subplot
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig('1_numerical_distributions.png', dpi=150)
    plt.close()
    print("Saved: 1_numerical_distributions.png")

def plot_categorical_frequencies(df):
    print("\n[4/7] Generating Categorical Frequency Charts...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Industry Sectors
    sns.countplot(y='Industry_Sectors', data=df, ax=axes[0], order=df['Industry_Sectors'].value_counts().index, palette='viridis')
    axes[0].set_title('Frequency of Industry Sectors', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Count')
    
    # Supply Chain Transport Mode
    sns.countplot(x='Supply_Chain_Transport_Mode', data=df, ax=axes[1], order=df['Supply_Chain_Transport_Mode'].value_counts().index, palette='muted')
    axes[1].set_title('Frequency of Transport Modes', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('2_categorical_frequencies.png', dpi=150)
    plt.close()
    print("Saved: 2_categorical_frequencies.png")

def plot_energy_trends(df):
    print("\n[5/7] Generating Energy Consumption Trends...")
    time_df = df.groupby('Date')[['Total_Energy_Consumption_kWh', 'Renewable_Energy_Consumption_kWh', 'NonRenewable_Energy_Consumption_kWh']].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_df['Date'], time_df['Total_Energy_Consumption_kWh'], label='Total Energy', color='#1A365D', linewidth=2)
    plt.plot(time_df['Date'], time_df['Renewable_Energy_Consumption_kWh'], label='Renewable Energy', color='#319795', linewidth=1.5)
    plt.plot(time_df['Date'], time_df['NonRenewable_Energy_Consumption_kWh'], label='Non-Renewable Energy', color='#E53E3E', linewidth=1.5)
    
    plt.title('Aggregate Energy Consumption Logs Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Timeline')
    plt.ylabel('Energy (kWh)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('3_energy_consumption_trends.png', dpi=150)
    plt.close()
    print("Saved: 3_energy_consumption_trends.png")

def bivariate_multivariate_analysis(df, numerical_cols):
    print("\n[6/7] Conducting Bivariate & Multivariate Analysis...")
    
    # 1. Scatter plots (Using Non-Renewable Energy as a core indicator since Temperature is absent)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sample_df = df.sample(n=min(1000, len(df)), random_state=42)
    
    sns.scatterplot(x='NonRenewable_Energy_Consumption_kWh', y='Carbon_Emission_tCO2e_TARGET', data=sample_df, alpha=0.6, ax=axes[0], color='#2B6CB0')
    axes[0].set_title('Emissions vs Non-Renewable Energy Consumption', fontsize=12, fontweight='bold')
    
    sns.scatterplot(x='Date', y='Carbon_Emission_tCO2e_TARGET', data=sample_df, alpha=0.6, ax=axes[1], color='#319795')
    axes[1].set_title('Emissions Output Over Time Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('4_scatter_analysis.png', dpi=150)
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr_cols = numerical_cols + ['Carbon_Emission_tCO2e_TARGET']
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Operational & Environmental Features', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('5_correlation_heatmap.png', dpi=150)
    plt.close()
    
    # 3. Boxplot grouped by Industry Sectors
    plt.figure(figsize=(14, 7))
    sns.boxplot(x='Carbon_Emission_tCO2e_TARGET', y='Industry_Sectors', data=df, palette='Set3')
    plt.title('Carbon Emissions Distribution Grouped by Industry Sectors', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('6_industry_emissions_boxplot.png', dpi=150)
    plt.close()
    
    # 4. Pairplot
    print("Generating Pairplot (this might take a moment)...")
    pairplot_features = ['Carbon_Emission_tCO2e_TARGET', 'Total_Energy_Consumption_kWh', 'Production_Output_Units']
    g = sns.pairplot(df[pairplot_features].sample(n=min(500, len(df)), random_state=42), diag_kind='kde')
    g.fig.suptitle('Pairplot Matrix of Key Operational Features', y=1.02, fontsize=14, fontweight='bold')
    g.savefig('7_features_pairplot.png', dpi=150)
    plt.close()
    print("Saved: 4_scatter_analysis.png, 5_correlation_heatmap.png, 6_industry_emissions_boxplot.png, 7_features_pairplot.png")

def time_series_exploration(df):
    print("\n[7/7] Executing Time Series Decompositions...")
    ts_df = df.groupby('Date')['Carbon_Emission_tCO2e_TARGET'].mean().reset_index()
    ts_df.set_index('Date', inplace=True)
    ts_df = ts_df.asfreq('D').fillna(method='ffill')
    
    # 1. Base Mean Daily Emissions Plot
    plt.figure(figsize=(14, 5))
    plt.plot(ts_df.index, ts_df['Carbon_Emission_tCO2e_TARGET'], color='#2B6CB0', label='Daily Avg Emissions')
    plt.title('Mean Daily Carbon Emissions Log', fontsize=13, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('tCO2e')
    plt.tight_layout()
    plt.savefig('8_time_series_base.png', dpi=150)
    plt.close()
    
    # 2. Decompose Trend and Seasonality
    decomp = seasonal_decompose(ts_df['Carbon_Emission_tCO2e_TARGET'], period=30)
    fig = decomp.plot()
    fig.set_size_inches(14, 10)
    plt.tight_layout()
    plt.savefig('9_time_series_decomposition.png', dpi=150)
    plt.close()
    
    # 3. Rolling Statistics
    plt.figure(figsize=(14, 6))
    plt.plot(ts_df.index, ts_df['Carbon_Emission_tCO2e_TARGET'], label='Original Daily Data', alpha=0.4, color='gray')
    plt.plot(ts_df.index, ts_df['Carbon_Emission_tCO2e_TARGET'].rolling(window=30).mean(), label='30-Day Moving Average', color='#2B6CB0', linewidth=2)
    plt.plot(ts_df.index, ts_df['Carbon_Emission_tCO2e_TARGET'].rolling(window=30).std(), label='30-Day Moving Std Dev', color='#E53E3E', linewidth=1.5)
    plt.title('Rolling Statistics & Variance Thresholds (30-Day Window)', fontsize=13, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('10_rolling_statistics.png', dpi=150)
    plt.close()
    print("Saved: 8_time_series_base.png, 9_time_series_decomposition.png, 10_rolling_statistics.png")

if __name__ == '__main__':
    # Define dataset filename
    DATASET_FILE = 'carbon_emission_dataset_with_Industry_bdd4e9a7e373f0c80481c4ff175f1cd9.csv'
    
    try:
        # Pipeline Execution
        dataframe = load_data(DATASET_FILE)
        numerical_columns = basic_statistics(dataframe)
        plot_distributions(dataframe, numerical_columns)
        plot_categorical_frequencies(dataframe)
        plot_energy_trends(dataframe)
        bivariate_multivariate_analysis(dataframe, numerical_columns)
        time_series_exploration(dataframe)
        
        print("\n=======================================================")
        print("DATA ANALYSIS PIPELINE COMPLETED SUCCESSFULLY!")
        print("All requested analytical plots have been generated and saved.")
        print("=======================================================")
        
    except Exception as e:
        print(f"\n[ERROR] An error occurred during execution: {e}")
