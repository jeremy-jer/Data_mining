import streamlit as st
import pandas as pd
import numpy as np

# Titre de l'application
st.title('Data Mining')
st.subheader('Analyse de données')
# Importation des données
df = pd.read_csv('data.csv')

# Affichage des données
st.write("Données importées :")
st.write(df)
st.write('-------------------------------------------------------------------------------------------------------------')

# Analyse des données
# Première ligne
st.write("5 premières lignes :")
st.write(df.head())
st.write('-------------------------------------------------------------------------------------------------------------')

# Dernière ligne
st.write("5 dernières lignes :")
st.write(df.tail())
st.write('-------------------------------------------------------------------------------------------------------------')

# Noms de colonnes
st.write("Noms des colonnes :")
st.write(df.columns)

# Nombre de lignes et de colonnes
st.write("Nombre de lignes et de colonnes :")
st.write(df.shape)

# Nombre de valeurs nulles
st.write("Nombre de valeurs nulles :")
st.write(df.isnull().sum())
