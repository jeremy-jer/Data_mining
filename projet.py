import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importation des données
df = pd.read_csv('data.csv')
print(df)
print('-------------------------------------------------------------------------------------------------------------')

# analyse des données
# premiere ligne
print("5 premieres ligne")
print(df.head())
print('-------------------------------------------------------------------------------------------------------------')

# derniere ligne
print("5 dernieres ligne")
print(df.tail())
print('-------------------------------------------------------------------------------------------------------------')

# noms de colonnes
print("nombre de colonnes")
print(df.columns)

# nombre de lignes et de colonnes
print("nombre de lignes et de colonnes")
print(df.shape)








