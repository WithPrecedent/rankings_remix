"""
rankings_remix: US News Law School Rankings Done Better
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

"""
from __future__ import annotations
import pathlib
from typing import (Any, Callable, ClassVar, Dict, Hashable, Iterable, List, 
    Mapping, MutableMapping, MutableSequence, Optional, Sequence, Set, Tuple, 
    Type, Union)
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing

DATA_FOLDER = pathlib.Path('..') / 'data' / 'rankings' 
VISUALS_FOLDER = pathlib.Path('..') / 'visuals' / 'rankings'
IMPORT_DATA_PATH = pathlib.Path(DATA_FOLDER) / '2022rankings.csv'
EXPORT_DATA_PATH = pathlib.Path(DATA_FOLDER) / 'remixed_rankings.csv'
 
RENAMES = { 
    'Rank': 'US News Rank',
    'Overall score': 'US News Score',
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
    'Propotion of 2020 J.D. graduates who borrowed at least one educational loan in law school': 'Student Percentage with Loan',	
    'Average indebtedness of 2020 J.D. graduates who incurred law school debt': 'Average Debt'}
TYPES = {
    'School': str,
    'US News Rank': int,
    'US News Score': int,
    'Peer Score': float,
    'Practitioner Score': float,
    'GPA': float,
    'LSAT': float,
    'Acceptance Rate': 'percent',
    'Student/Faculty Ratio': float,
    'Immediate Employed': 'percent',
    '10-Month Employed': 'percent',
    'Bar Pass Rate': 'percent',
    'Jurisdiction Bar Pass Rate': 'percent',
    'Bar Pass Ratio': float,
    'Student Percentage with Loan': 'percent',
    'Average Debt': int}
LOW_IS_BETTER = [
    'Acceptance Rate', 
    'Student/Faculty Ratio',
    'Student Percentage with Loan',
    'Average Debt']
USNEWS_WEIGHTS = {
    'Peer Score': 0.25,
    'Practitioner Score': 0.15,
    'GPA Estimated Median': 0.0875,
    'LSAT Estimated Median': 0.1125,
    'Acceptance Rate': .01,
    'Immediate Employed': 0.04,
    '10-Month Employed': 0.14,
    'Bar Pass Ratio': 0.0225,
    'Student/Faculty Ratio': 0.02,
    'Student Percentage with Loan': 0.02,
    'Average Debt': 0.03}
CORE_COLUMNS = [
    'Peer Score',
    'Practitioner Score',
    'GPA Estimated Median',
    'LSAT Estimated Median',
    '10-Month Employed',
    'Average Debt']    

def minmax_scale(data: pd.Series, low_is_good: bool = False) -> pd.Series:
    if low_is_good:
        data = pd.Series(1 - data)
    return preprocessing.minmax_scale(data)

def standard_scale(data: pd.Series, low_is_good: bool = False) -> pd.Series:
    if low_is_good:
        data = pd.Series(1 - data)
    return preprocessing.scale(data)

scalers = {
    'Percent': minmax_scale,
    'Standardized': standard_scale}
    
def import_rankings_data() -> pd.DataFrame:
    df = pd.read_csv(IMPORT_DATA_PATH, encoding = 'windows-1252')
    df = df.rename(columns = RENAMES)
    return df
    
def fix_lower_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df['US News Rank'] = df['US News Rank'].str.split('-').str[0]
    df['US News Rank'] = pd.to_numeric(df['US News Rank'], downcast = 'integer')
    df['US News Score Scaled'] = minmax_scale(data = df['US News Score'])
    df['US News Rank Scaled'] = minmax_scale(data = df['US News Rank'], 
                                             low_is_good = True)
    return df

def change_to_numerical(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        for suffix, kind in TYPES.items():
            if column.endswith(suffix) and df.dtypes[column] in [object]:
                if kind in [int]:
                    df[column] = df[column].str.replace('$', '')
                    df[column] = df[column].str.replace(',', '')
                    df[column] = df[column].astype('float')
                    df[column] = pd.to_numeric(df[column], downcast = 'integer')
                elif suffix in ['GPA', 'LSAT']:
                    low = f'{column} 25'
                    high = f'{column} 75'
                    median = f'{column} Estimated Median'
                    df[[low, high]] = df[column].str.split('-', expand = True)
                    df[low] = pd.to_numeric(df[low])
                    df[high] = pd.to_numeric(df[high])
                    df[median] = df[[low, high]].mean(axis = 1)
                elif kind in [float, 'percent']:
                    df[column] = (df[column].str.split('%')
                                            .str[0]
                                            .astype('float'))
                    if kind in ['percent']:
                        df[column] = df[column]/100
    return df

def compute_bar_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df['Bar Pass Ratio'] = (
        df['Bar Pass Rate'] / df['Jurisdiction Bar Pass Rate'])
    return df

def ordinal_rank(df: pd.DataFrame, 
                 column: str, 
                 method: str = 'max') -> pd.Series:
    return df[column].rank(method = method)

def scale_all_usnews_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name, scaler in scalers.items():
        for column in USNEWS_WEIGHTS.keys():
            kwargs = {}
            if column in LOW_IS_BETTER:
                kwargs['low_is_good'] = True
            scaled_column = f'{name} {column}'
            df[scaled_column] = scaler(data = df[column], **kwargs)
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
        df[scaled_score_column] = minmax_scale(data = df[score_column])
    core_columns = [col for col in df if col.startswith('Percent')]
    core_columns = [
        col for col in core_columns if col.endswith(tuple(CORE_COLUMNS))]
    core_weights = {k: v for k, v in USNEWS_WEIGHTS.items() if k in CORE_COLUMNS}
    core_weights = list(core_weights.values())
    df['Core Score'] = df[core_columns].mul(core_weights).sum(1)
    df['Core Score Scaled'] = minmax_scale(data = df['Core Score'])
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
    df['US News Rank Adjusted'] = df['US News Rank'].rank(method = 'min', 
                                                          ascending = True) 
    return df

def add_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    df['Hidden Data Rank Boost'] = (df['Standardized Rank'] 
                                    - df['US News Rank Adjusted'])
    df['Standardization Rank Boost'] = (df['Percent Rank'] 
                                        - df['Standardized Rank'])
    df['Questionable Data Categories Rank Boost'] = (df['Core Rank'] 
                                                     - df['Percent Rank'])
    df['Hidden Data Score Boost'] = (df['US News Score Scaled'] 
                                     - df['Standardized Score Scaled']) 
    df['Standardization Score Boost'] = (df['Standardized Score Scaled'] 
                                         - df['Percent Score Scaled'])
    df['Questionable Data Categories Score Boost'] = (df['Percent Score Scaled'] 
                                                      - df['Core Score Scaled']) 
    df['Total Rank Boost'] = (df['Hidden Data Rank Boost']
                              + df['Standardization Rank Boost']
                              + df['Questionable Data Categories Rank Boost'])
    df['Total Score Boost'] = (df['Hidden Data Score Boost']
                               + df['Standardization Score Boost']
                               + df['Questionable Data Categories Score Boost'])
    return df

def visualize_rank_versus_score(df: pd.DataFrame) -> None:
    rank_score = sns.scatterplot(x = df['US News Score'], 
                                 y = df['US News Rank Adjusted'], 
                                 color = 'aqua')   
    rank_score.set_xlim([df['US News Score'].min(), df['US News Score'].max()])
    rank_score.set_ylim([df['US News Rank Adjusted'].min(), 
                         df['US News Rank Adjusted'].max()])
    rank_score.figure.suptitle(
        'US News Rank vs. US News Score', 
        size = 16)
    plt.tight_layout()
    export_path = pathlib.Path(VISUALS_FOLDER) / 'rank_score.png'
    rank_score.figure.savefig(export_path)
    plt.close() 
    return
   
def visualize_standardization_effects(df: pd.DataFrame) -> None:
    percent_columns = [f'Percent {key}' for key in USNEWS_WEIGHTS]
    standardized_columns = [f'Standardized {key}' for key in USNEWS_WEIGHTS]
    column_groups = [percent_columns, standardized_columns]
    distributions, axes = plt.subplots(figsize = (12, 6), 
                                       ncols = len(column_groups), 
                                       sharey = True)
    distributions.suptitle(
        'Effects of Standardization on Component Score Distributions', 
        size = 16)
    for i, column_group in enumerate(column_groups):
        for column in column_group:
            distributions = sns.kdeplot(df[column], 
                                        ax = axes[i], 
                                        legend = False)
    axes[0].set_xlim([0, 1])
    axes[1].set_xlim([-3, 3])
    distributions.figure.legend(title = 'Categories', 
               bbox_to_anchor = (1.01, 1), 
               borderaxespad = 0, 
               labels = list(USNEWS_WEIGHTS.keys()), 
               prop = {'size': 10})
    plt.tight_layout()
    export_path = pathlib.Path(VISUALS_FOLDER) / 'category_distributions.png'
    distributions.figure.savefig(export_path)
    plt.close() 
    return
  
def visualize_score_rank_distributions(df: pd.DataFrame) -> None:
    distributions, axis = plt.subplots()
    distributions.suptitle('Distributions of US News Scores and Rankings', 
                           size = 16)
    sns.histplot(df['US News Score Scaled'], 
                 ax = axis, 
                 color = 'blue', 
                 kde = True, 
                 label = 'US News Score')
    sns.histplot(df['US News Rank Scaled'], 
                 ax = axis, 
                 color = 'orange', 
                 kde = True, 
                 label = 'US News Rank')
    axis.set(xlabel = 'US News Scores and Ranks (percent scale)')
    axis.set_xlim([0, 1])
    plt.legend()
    plt.tight_layout()
    export_path = pathlib.Path(VISUALS_FOLDER) / 'score_rank_distributions.png'
    distributions.figure.savefig(export_path)
    plt.close()
    return

def visualize_comparisons(df: pd.DataFrame) -> None:
    public_scatter = sns.scatterplot(x = df['Standardized Rank'], 
                                     y = df['US News Rank'], 
                                     color = 'green')
    plt.xlabel('Estimated US News Rank Based on Public Data')
    public_scatter.set_xlim([df['US News Rank'].min(), 
                             df['US News Rank'].max()])
    public_scatter.set_ylim([df['US News Rank'].min(), 
                             df['US News Rank'].max()])
    public_scatter.figure.suptitle(
        'Effects of Hidden Data on School Rankings', 
        size = 16)
    plt.tight_layout()
    export_path = pathlib.Path(VISUALS_FOLDER) / 'public_scatter.png'
    public_scatter.figure.savefig(export_path)
    plt.close()
    score_comparison = sns.scatterplot(x = df['US News Score Scaled'], 
                                       y = df['US News Score'], 
                                       label = 'US News', 
                                       color = 'orange', 
                                       alpha = 0.35)
    score_comparison = sns.scatterplot(x = df['Percent Score Scaled'], 
                                       y = df['US News Score'], 
                                       label = 'Percent', 
                                       color = 'olive', 
                                       alpha = 0.5)
    score_comparison = sns.scatterplot(x = df['Core Score Scaled'], 
                                       y = df['US News Score'], 
                                       label = 'Core Metrics', 
                                       color = 'purple', 
                                       alpha = 0.35)
    score_comparison.set_xlim(0, 1)
    score_comparison.set_ylim([df['US News Score'].min() - 5, 
                               df['US News Score'].max() + 5])
    plt.xlabel('Percent Score')
    score_comparison.figure.suptitle(
        'Comparison of Different Scoring Methodologies', 
        size = 16)
    plt.tight_layout()
    export_path = pathlib.Path(VISUALS_FOLDER) / 'score_comparison.png'
    score_comparison.figure.savefig(export_path)
    plt.close()
    return

def export_remixed_rankings(df: pd.DataFrame) -> None:
    df.to_csv(EXPORT_DATA_PATH)
    return
        
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    pd.set_option('precision', 0)
    sns.set_style('whitegrid')
    complete_df = import_rankings_data()
    complete_df = fix_lower_ranks(df = complete_df)
    only_ranked_df = complete_df[complete_df['US News Rank'] < 145]
    check_na_columns = list(RENAMES.values())
    check_na_columns = check_na_columns.remove('US News Score')
    df = complete_df.dropna(subset = check_na_columns)
    df = change_to_numerical(df = df)
    df = compute_bar_ratio(df = df)
    df = scale_all_usnews_columns(df = df)
    df = compute_scores(df = df)
    df = compute_ranks(df = df)
    df = add_comparisons(df = df)
    # visualize_rank_versus_score(df = df)
    visualize_standardization_effects(df = df)
    visualize_score_rank_distributions(df = only_ranked_df)
    complete_data_df = add_comparisons(df = df)
    visualize_comparisons(df = df)
    export_remixed_rankings(df = df)
    