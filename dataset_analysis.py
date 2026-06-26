import pandas as pd
import os

dataset = pd.read_csv('synthetic_knowledge_items.csv')
print(dataset.head())

# removing null values from the dataset
print("Null values checker:", dataset.isnull().sum())
dataset = dataset.dropna()
print("Null values after removal:", dataset.isnull().sum())


