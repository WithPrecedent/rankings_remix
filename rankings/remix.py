"""
rankings_remix: US News Law School Rankings Done Better
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

"""
from __future__ import annotations
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FOLDER = pathlib.Path('..') / 'data' / 'rankings' 
IMPORT_DATA_PATH = pathlib.Path(DATA_FOLDER) / '2022rankings.csv'
EXPORT_DATA_PATH = pathlib.Path(DATA_FOLDER) / 'remixed_rankings.csv'
VISUALS_FOLDER = pathlib.Path('..') / 'visuals' / 'rankings' 
RENAMES = { 
    'Rank': 'USNews Rank',
    'Overall score': 'USNews Score',
    'Peer assessment score (5.0=highest)': 'Peer Score',
    'Assessment score by lawyers/judges (5.0=highest)': 'Practitioner Score',
    '2020 undergrad GPA 25th-75th percentile': 'GPA',
    '2020 LSAT score 25th-75th percentile': 'LSAT',
    '2020 acceptance rate': 'Acceptance Rate',
    '2020 student/faculty ratio': 'Student/Faculty Ratio',
    '2019 grads employed at graduation²': 'Immediate Employed',
    '2019 grads employed 10 months after graduation²': '10-Month Employed',
    'School\'s bar passage rate in jurisdiction': 'Bar Pass Rate',
    'Jurisdiction\'s overall bar passage rate': 'Jurisdiction Bar Pass Rate',
    'Propotion of 2020 J.D. graduates who borrowed at least one educational loan in law school': 'Percent with Loan',	
    'Average indebtedness of 2020 J.D. graduates who incurred law school debt': 'Average Debt'}
FLOATS = [
    'USNews Score',
    'Peer Score',
    'Practitioner Score',
    'GPA 25',
    'GPA 75',
    'GPA Estimated Median',
    'Acceptance Rate',
    'Student/Faculty Ratio',
    'Immediate Employed',
    '10-Month Employed',
    'Bar Pass Rate',
    'Jurisdiction Bar Pass Rate',
    'Percent with Loan']
INTEGERS = [
    'USNews Rank',
    'LSAT 25',
    'LSAT 75',
    'LSAT Estimated Median',
    'Average Debt']
LOW_IS_HIGH = ['Acceptance Rate', 'Student/Faculty Ratio']
USNEWS_WEIGHTS = {
    'Peer Score': 0.25,
    'Practitioner Score': 0.15,
    'GPA Estimated Median': 0.1,
    'LSAT Estimated Median': 0.125,
    'Acceptance Rate': .025,
    'Immediate Employed': 0.04,
    '10-Month Employed': 0.14,
    'Bar Pass Ratio': 0.0225,
    'Student/Faculty Ratio': 0.02}
CORE_COLUMNS = [
    'Peer Score',
    'Practitioner Score',
    'GPA Estimated Median',
    'LSAT Estimated Median',
    '10-Month Employed']    
RANK_COMPARISON_COLUMNS = [
    # 'USNews Rank',
    # 'School',
    'Hidden Data Rank Boost',
    'Aggregation Method Rank Boost',
    'Questionable Data Categories Rank Boost']
SCORE_COMPARISON_COLUMNS = [
    # 'USNews Rank',
    # 'School',
    # 'USNews Score',
    'Hidden Data Score Boost',
    'Aggregation Method Score Boost',
    'Questionable Data Categories Score Boost']

def import_rankings_data() -> pd.DataFrame:
    df = pd.read_csv(IMPORT_DATA_PATH, encoding = 'windows-1252')
    df = df.rename(columns = RENAMES)
    df = df[~df.eq('N/A').any(1)]
    df = df.dropna()
    return df

def simplify_lower_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df['USNews Rank'] = df['USNews Rank'].str.split('-').str[0]
    df['USNews Rank'] = pd.to_numeric(df['USNews Rank'], downcast = 'integer')
    df = df[df['USNews Rank'] < 112]
    return df
    
def split_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ['GPA', 'LSAT']:
        low = f'{column} 25'
        high = f'{column} 75'
        median = f'{column} Estimated Median'
        df[[low, high]] = df[column].str.split('-', expand = True)
        df[low] = pd.to_numeric(df[low])
        df[high] = pd.to_numeric(df[high])
        df[median] = df[[low, high]].mean(axis = 1)
    return df

def force_numerical(df: pd.DataFrame) -> pd.DataFrame:
    for column in FLOATS:
        if df.dtypes[column] in [object]:
            df[column] = df[column].str.split('%').str[0].astype('float')/100
    for column in INTEGERS:
        if df.dtypes[column] in [object]:
            df[column] = df[column].str.replace('$', '')
            df[column] = df[column].str.replace(',', '')
            df[column] = df[column].astype('float')
            df[column] = pd.to_numeric(df[column], downcast = 'integer')
    return df

def compute_bar_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['Bar Pass Ratio'] = (
        df['Bar Pass Rate'] / df['Jurisdiction Bar Pass Rate'])
    return df

def reverse_values(df: pd.DataFrame) -> pd.DataFrame:
    for column in LOW_IS_HIGH:
        df[column] = 1 - df[column]
    return df

def minmax_scale(df: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
    df[destination] = (df[source] - df[source].min()) / (df[source].max() - df[source].min())
    return df

def ordinal_scale(df: pd.DataFrame, source: str, destination: str) -> pd.DataFrame:
    df[destination] = df[source].rank(method = 'max')
    return df

scalers = {
    'Percentile': minmax_scale,
    'Ordinal': ordinal_scale}
    
def scale_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name, scaler in scalers.items():
        for column in USNEWS_WEIGHTS.keys():
            scaled_column = f'{name} {column}'
            df = scaler(df = df, source = column, destination = scaled_column)
    return df 

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:   
    keys = tuple(USNEWS_WEIGHTS.keys())
    for name in scalers.keys():
        columns = [col for col in df if col.startswith(name)]
        columns = [col for col in columns if col.endswith(keys)]
        score_column = f'{name} Score'
        scaled_score_column = f'{score_column} Scaled'
        weights = list(USNEWS_WEIGHTS.values())
        df[score_column] = df[columns].mul(weights).sum(1)
        df = minmax_scale(df = df, source = score_column, destination = scaled_score_column)
    core_columns = [col for col in df if col.startswith('Percentile')]
    core_columns = [col for col in core_columns if col.endswith(tuple(CORE_COLUMNS))]
    core_weights = {k: v for k, v in USNEWS_WEIGHTS.items() if k in CORE_COLUMNS}
    core_weights = list(core_weights.values())
    df['Core Score'] = df[core_columns].mul(core_weights).sum(1)
    df = minmax_scale(df = df, source = 'Core Score', destination = 'Core Score Scaled')
    df = minmax_scale(df = df, source = 'USNews Score', destination = 'USNews Score Scaled')
    df = minmax_scale(df = df, source = 'USNews Rank', destination = 'USNews Rank Scaled')
    return df

def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    keys = tuple(USNEWS_WEIGHTS.keys())
    for name in scalers.keys():
        columns = [col for col in df if col.startswith(name)]
        columns = [col for col in columns if col.endswith(keys)]
        score_column = f'{name} Score'
        rank_column = f'{name} Rank'
        df[rank_column] = df[score_column].rank(method = 'min', ascending = False)
    df['Core Rank'] = df['Core Score'].rank(method = 'min', ascending = False) 
    return df

def add_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    df['Hidden Data Rank Boost'] = df['Ordinal Rank'] - df['USNews Rank']
    df['Aggregation Method Rank Boost'] = df['Percentile Rank'] - df['USNews Rank']
    df['Questionable Data Categories Rank Boost'] = df['Core Rank'] - df['USNews Rank']
    df['Hidden Data Score Boost'] = df['USNews Score Scaled'] - df['Ordinal Score Scaled'] 
    df['Aggregation Method Score Boost'] = df['USNews Score Scaled'] - df['Percentile Score Scaled'] 
    df['Questionable Data Categories Score Boost'] = df['USNews Score Scaled'] - df['Core Score Scaled'] 
    return df
  
def visualize_distributions(df: pd.DataFrame) -> None:
    distributions, axis = plt.subplots()
    sns.distplot(df['USNews Score Scaled'], ax = axis, color = 'blue', label = 'USNews Score')
    sns.distplot(df['USNews Rank Scaled'], ax = axis, color = 'orange', label = 'USNews Rank', axlabel = 'USNews Scores and Ranks (Common Scale)')
    axis.legend()
    axis.set_xlim([0, 1])
    export_path = pathlib.Path(VISUALS_FOLDER) / 'distributions.png'
    distributions.savefig(export_path)
    plt.close()
    return

def visualize_comparisons(df: pd.DataFrame) -> None:
    public_scatter = sns.scatterplot(x = df['Ordinal Rank'], y = df['USNews Rank'], color = 'green')
    plt.ylabel('USNews Public Data Rank')
    public_scatter.set_xlim([df['USNews Rank'].min(), df['USNews Rank'].max()])
    public_scatter.set_ylim([df['USNews Rank'].min(), df['USNews Rank'].max()])
    export_path = pathlib.Path(VISUALS_FOLDER) / 'public_scatter.png'
    public_scatter.figure.savefig(export_path)
    plt.close()
    score_comparison = sns.scatterplot(x = df['Percentile Score Scaled'], y = df['USNews Score'], label = 'Percentile', color = 'olive', alpha = 0.5)
    score_comparison = sns.scatterplot(x = df['Core Score Scaled'], y = df['USNews Score'], label = 'Core Metrics', color = 'purple', alpha = 0.35)
    score_comparison.set_xlim(0, 1)
    score_comparison.set_ylim([df['USNews Score'].min() - 5, df['USNews Score'].max() + 5])
    plt.xlabel('Alternative Scores')
    export_path = pathlib.Path(VISUALS_FOLDER) / 'score_comparison.png'
    score_comparison.figure.savefig(export_path)
    plt.close()
    return

def rank_comparison_table(df: pd.DataFrame) -> None:
    rank_data = df[RANK_COMPARISON_COLUMNS]
    sns.heatmap(rank_data, annot = True, yticklabels = df['School'])
    plt.show()
    return
    
def export_remixed_rankings(df: pd.DataFrame) -> None:
    df.to_csv(EXPORT_DATA_PATH)
    return
        
if __name__ == '__main__':
    pd.set_option('precision', 0)
    sns.set_style('whitegrid')
    df = import_rankings_data()
    df = simplify_lower_ranks(df = df)
    df = split_columns(df = df)
    df = force_numerical(df = df)
    df = compute_bar_ratio(df = df)
    df = reverse_values(df = df)
    df = scale_columns(df = df)
    df = compute_scores(df = df)
    df = compute_ranks(df = df)
    df = add_comparisons(df = df)
    visualize_distributions(df = df)
    visualize_comparisons(df = df)
    rank_comparison_table(df = df)
    export_remixed_rankings(df = df)
    