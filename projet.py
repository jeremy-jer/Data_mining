import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importation des données
df = pd.read_csv('data.csv')
print(df)
print('-------------------------------------------------------------------------------------------------------------')

# analyse des données
print(df.describe())
print('-------------------------------------------------------------------------------------------------------------')

print(df.info())

print('-------------------------------------------------------------------------------------------------------------')



