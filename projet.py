import streamlit as st
import pandas as pd
from sklearn.impute import KNNImputer

def convert_columns(df):
    for col in df.columns:
        # Essayez de convertir les colonnes numériques qui sont lues comme des objets en int ou float
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    return df

def main():
    st.title("Data mining")

    # Section pour charger le fichier CSV
    st.subheader("Chargement des données")
    file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

    st.sidebar.header('Menu')
    partie = st.sidebar.radio("Titre des parties", ('Data Pre-processing and Cleaning', 'Visualization of the cleaned data', 'Clustering or Prediction', 'Learning Evaluation'))    
    
    df = None 
    
    # Chargement des données si un fichier est téléchargé
    if file is not None:
        try:
            df = pd.read_csv(file, on_bad_lines='skip', sep=';')
            st.success("Fichier chargé avec succès")
            df = convert_columns(df)
        except pd.errors.ParserError as e:
            st.error(f"Erreur lors de la lecture du fichier: {e}")
        except Exception as e:
            st.error(f"Une erreur est survenue: {e}")
    
    if df is not None:
        if partie == 'Data Pre-processing and Cleaning':
            st.subheader("Aperçu des données")
            st.markdown("*Premières lignes du jeu de données :*")
            st.write(df.head())
            st.markdown("*Dernières lignes du jeu de données :*")
            st.write(df.tail())

            st.subheader("Étude statistique des données")
            st.write("Nombre total de lignes :", df.shape[0])
            st.write("Nombre total de colonnes :", df.shape[1])
            st.write("Liste des colonnes :", df.columns.tolist())
            st.write("Nombre de valeurs manquantes par colonne :")
            st.write(df.isnull().sum())

            st.write("Description de la moyenne, la médiane, l'écart-type.... ")
            st.write(df.describe())

        elif partie == 'Visualization of the cleaned data':
            st.write("Quelle opération voulez-vous faire ? ")
            choix = st.selectbox('Choix', ['Supprimer les lignes', 'Supprimer les colonnes', 'Remplacer les valeurs nulles', 'Imputer par KNN'])

            if choix == 'Supprimer les lignes':
                st.write("Suppression de lignes")
                df = df.dropna()
                st.write(df.head())
            elif choix == 'Supprimer les colonnes':
                st.write("Suppression de colonnes")
                df = df.dropna(axis=1)
                st.write(df.head())
            elif choix == 'Remplacer les valeurs nulles par la moyenne':
                st.write("Remplacement des valeurs nulles")
                df = df.fillna(df.mean())
                st.write(df.head())
            elif choix == 'Remplacer par KNN':
                st.write("KNN")
                Knn = KNNImputer(n_neighbors=5)
                df[:] = Knn.fit_transform(df)
                st.write(df.head())
                st.write(df.isnull().sum())

if __name__ == "__main__":
    main()
