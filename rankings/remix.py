"""
rankings_remix: US News Law School Rankings Done Better
Corey Rayburn Yung <coreyrayburnyung@gmail.com>
Copyright 2020-2021, Corey Rayburn Yung
License: Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)

"""
from __future__ import annotations
import dataclasses
import itertools
import pathlib
from typing import (Any, Callable, ClassVar, Dict, Hashable, Iterable, List, 
    Mapping, MutableMapping, MutableSequence, Optional, Sequence, Set, Tuple, 
    Type, Union)

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import preprocessing
import seaborn as sn

FOLDER = pathlib.Path('..') / 'data' / 'rankings' 
IMPORTPATH = pathlib.Path(FOLDER) /'2022rankings.csv'
EXPORTPATH = pathlib.Path(FOLDER) / 'remixed_rankings.csv'
RENAMES = { 
    'Rank': 'USNews Rank',
    'Overall score': 'USNews Score',
    'Peer assessment score (5.0=highest)': 'Peer Score',
    'Assessment score by lawyers/judges (5.0=highest)': 'Practitioner Score',
    '2020 undergrad GPA 25th-75th percentile': 'GPA',
    '2020 LSAT score 25th-75th percentile': 'LSAT',
    '2020 acceptance rate': 'Acceptance Rate',
    '2020 student/faculty ratio': 'Student/Faculty Ratio',
    '2019 grads employed at graduationý': 'Immediate Employed',
    '2019 grads employed 10 months after graduationý': '10-Month Employed',
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
    # 'Bar Pass Ratio',
    'Percent with Loan']
INTEGERS = [
    'USNews Rank',
    'LSAT 25',
    'LSAT 75',
    'LSAT Estimated Median',
    'Average Debt']
USNEWSWEIGHTS = {
    'Peer Score': 0.25,
    'Practitioner Score': 0.15,
    'GPA Estimated Median': 0.1,
    'LSAT Estimate Median': 0.125,
    'Acceptance Rate': .025,
    'Immediate Employed': 0.04,
    '10-Month Employed': 0.14,
    'Bar Pass Ratio': 0.02}
SCALERS = {
    'MinMax': preprocessing.MinMaxScaler,
    'Ordinal': preprocessing.OrdinalEncoder,
    'Normalized': preprocessing.Normalizer}
FINAL_COLUMNS = [
    'School',
    'Peer Score',
    'Practitioner Score',
    'GPA Estimated Median',
    'LSAT Estimate Median',
    'Acceptance Rate',
    'Immediate Employed',
    '10-Month Employed',
    'Bar Pass Ratio',
    'USNews Rank',
    'Adjusted USNews Score',
    'Normalized Rank',
    'Normalized Score',
    'MinMax Rank',
    'MinMax Score',
    'Ordinal Rank',
    'Ordinal Score'
]
def import_rankings_data() -> pd.DataFrame:
    df = pd.read_csv(IMPORTPATH, encoding = 'windows-1252')
    df = df.rename(columns = RENAMES)
    df = df[~df.eq('N/A').any(1)]
    df = df.dropna()
    return df

def simplify_lower_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df['USNews Rank'] = df['USNews Rank'].str.split('-').str[0]
    return df
    
def split_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in ['GPA', 'LSAT']:
        low = f'{column} 25'
        high = f'{column} 75'
        median = f'{column} Estimated Median'
        df[[low, high]] = df[column].str.split('-', expand = True)
        df[low] = pd.to_numeric(df[low])
        df[high] = pd.to_numeric(df[high])
        df[f'{column} Estimated Median'] = df[[low, high]].mean(axis = 1)
        if column in ['LSAT']:
            df[median] = df[median].round()
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
    df['Bar Passage Ratio'] = (
        df['Bar Pass Rate'] / df['Jurisdiction Bar Pass Rate'])
    return df
    
def export_remixed_rankings(df: pd.DataFrame) -> None:
    df.to_csv(EXPORTPATH)
    return

def scale_columns(df: pd.DataFrame) -> pd.DataFrame:
    for name, scaler in SCALERS.items():
        print('test name scaler', name)
        new_columns = []
        scaler = scaler()
        for column in USNEWSWEIGHTS.keys():
            scaled_column = f'{name} {column}'
            new_columns.append(scaled_column)
            scaler.fit(df[column])
            df[scaled_column] = scaler.transform(df[column])
        score_column = f'{name} Score'
        rank_column = f'{name} Rank'
        weights = list(USNEWSWEIGHTS.values())
        df[score_column] = df[new_columns].mul(weights).sum(1)
        ranker = preprocessing.OrdinalEncoder()
        ranker.fit(df[score_column])
        df[rank_column] = ranker.transform(df[score_column])
        return df 
            
if __name__ == '__main__':
    pd.set_option('precision', 0)
    df = import_rankings_data()
    df = simplify_lower_ranks(df = df)
    df = split_columns(df = df)
    df = force_numerical(df = df)
    df = compute_bar_ratio(df = df)
    df = scale_columns(df = df)
    export_remixed_rankings(df = df)
    

