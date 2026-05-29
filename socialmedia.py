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

quality_counts = df['quality'].value_counts()

plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts, color='deeppink')
plt.title('Count Plot of Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()