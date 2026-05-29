import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings as wr
import seaborn as sns

df = pd.read_csv('data.csv')
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
df.select_dtypes(include='object').columns
df.select_dtypes(exclude='object').columns
