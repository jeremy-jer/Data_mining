from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import numpy as np

def convert_columns(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass
    return df

def main():
    st.title("Data Mining Application")

    st.subheader("Chargement des données")

    delimiter = st.text_input("Spécifiez le délimiteur pour le fichier CSV (par défaut ;)", value=';')
    file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

    st.sidebar.header('Menu')
    partie = st.sidebar.radio("Titre des parties", ('Data Pre-processing and Cleaning', 'Visualization of the cleaned data', 'Clustering or Prediction', 'Learning Evaluation'))

    df = None

    if file is not None:
        try:
            df = pd.read_csv(file, on_bad_lines='skip', sep=delimiter, low_memory=False)
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

            st.write("Quelle opération voulez-vous faire ? ")
            choix = st.selectbox('Choix', ['Supprimer les lignes', 'Supprimer les colonnes', 'Remplacer les valeurs nulles', 'Imputer par KNN', 'Normalisation des données'])

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

            elif choix == 'Imputer par KNN':
                st.write("Imputation par KNN")
                try:
                    numeric_df = df.select_dtypes(include=[int, float])
                    non_numeric_df = df.select_dtypes(exclude=[int, float])

                    if numeric_df.isnull().sum().sum() == 0:
                        st.write("Il n'y a pas de valeurs manquantes dans les colonnes numériques.")
                    else:
                        Knn = KNNImputer(n_neighbors=5)
                        df_imputed = pd.DataFrame(Knn.fit_transform(numeric_df), columns=numeric_df.columns)
                        df = pd.concat([df_imputed, non_numeric_df], axis=1)
                        st.write(df.head())
                except Exception as e:
                    st.error(f"Erreur lors de l'imputation KNN: {e}")

            elif choix == 'Normalisation des données':
                st.write("Choisir une méthode de normalisation")
                method = st.selectbox('Méthode de normalisation', ['Min-Max', 'Z-score', 'Decimal scaling', 'Feature Scaling'])

                try:
                    numeric_df = df.select_dtypes(include=[int, float])
                    non_numeric_df = df.select_dtypes(exclude=[int, float])

                    if method == 'Min-Max':
                        scaler = MinMaxScaler()
                    elif method == 'Z-score':
                        scaler = StandardScaler()
                    elif method == 'Decimal scaling':
                        scaler = RobustScaler()
                    elif method == 'Feature Scaling':
                        scaler = Normalizer()

                    scaled_data = scaler.fit_transform(numeric_df)
                    df_scaled = pd.DataFrame(scaled_data, columns=numeric_df.columns)
                    df = pd.concat([df_scaled, non_numeric_df], axis=1)
                    st.write(df.head())
                except Exception as e:
                    st.error(f"Erreur lors de la normalisation des données: {e}")

        elif partie == 'Visualization of the cleaned data':
            type_graphe = st.selectbox("Quel graphe vous voulez utiliser pour analyser vos courbes", ['Histogrammes', 'Box plot'])

            if type_graphe == 'Histogrammes':
                st.subheader("Histogrammes à variables")
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        fig = px.histogram(df, x=col, nbins=30, title=f'Histogramme de {col}')
                        st.plotly_chart(fig)

            elif type_graphe == 'Box plot':
                st.subheader("Box plots")
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        fig = px.box(df, y=col, title=f'Box plot de {col}')
                        st.plotly_chart(fig)

        elif partie == 'Clustering or Prediction':
            task = st.selectbox("Sélectionnez une tâche", ['Clustering', 'Prediction'])

            if task == 'Clustering':
                st.subheader("Clustering")
                algorithm = st.selectbox("Sélectionnez un algorithme de clustering", ['K-means', 'DBSCAN'])

                if algorithm == 'K-means':
                    n_clusters = st.number_input("Nombre de clusters", min_value=1, value=3)
                    kmeans = KMeans(n_clusters=n_clusters)

                    imputer = SimpleImputer(strategy='mean')
                    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[int, float])), columns=df.select_dtypes(include=[int, float]).columns)

                    df['Cluster'] = kmeans.fit_predict(df_imputed)
                    st.write(df.head())
                    fig = px.scatter_matrix(df, dimensions=df_imputed.columns, color='Cluster')
                    st.plotly_chart(fig)

                elif algorithm == 'DBSCAN':
                    eps = st.number_input("Valeur de epsilon", min_value=0.1, value=0.5)
                    min_samples = st.number_input("Nombre minimum d'échantillons", min_value=1, value=5)
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)


                    imputer = SimpleImputer(strategy='mean')
                    df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[int, float])), columns=df.select_dtypes(include=[int, float]).columns)

                    df['Cluster'] = dbscan.fit_predict(df_imputed)
                    st.write(df.head())
                    fig = px.scatter_matrix(df, dimensions=df_imputed.columns, color='Cluster')
                    st.plotly_chart(fig)

            elif task == 'Prediction':
                st.subheader("Prediction")
                problem_type = st.selectbox("Sélectionnez un type de problème", ['Regression', 'Classification'])

                target = st.selectbox("Sélectionnez la variable cible", df.columns)

                if problem_type == 'Regression':
                    algorithm = st.selectbox("Sélectionnez un algorithme de régression", ['Linear Regression', 'Random Forest Regressor'])

                    X = df.drop(columns=[target]).select_dtypes(include=[int, float])
                    y = df[target]

                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

                    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

                    if algorithm == 'Linear Regression':
                        model = LinearRegression()
                    elif algorithm == 'Random Forest Regressor':
                        model = RandomForestRegressor()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

                elif problem_type == 'Classification':
                    algorithm = st.selectbox("Sélectionnez un algorithme de classification", ['Logistic Regression', 'Random Forest Classifier'])

                    X = df.drop(columns=[target]).select_dtypes(include=[int, float])
                    y = df[target]

                    imputer = SimpleImputer(strategy='mean')
                    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

                    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

                    if algorithm == 'Logistic Regression':
                        model = LogisticRegression(max_iter=1000)
                    elif algorithm == 'Random Forest Classifier':
                        model = RandomForestClassifier()

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        elif partie == 'Learning Evaluation':
            st.subheader("Learning Evaluation")

            clustering_algo = st.selectbox("Choisissez l'algorithme de clustering à évaluer", ['K-means', 'DBSCAN'])

            if clustering_algo == 'K-means':
                n_clusters = st.number_input("Nombre de clusters", min_value=1, value=3)
                kmeans = KMeans(n_clusters=n_clusters)
                imputer = SimpleImputer(strategy='mean')
                df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[int, float])), columns=df.select_dtypes(include=[int, float]).columns)
                clusters = kmeans.fit_predict(df_imputed)
                df['Cluster'] = clusters

                fig = px.scatter_matrix(df, dimensions=df_imputed.columns, color='Cluster', title="Visualisation des clusters K-means")
                st.plotly_chart(fig)

                st.write("Statistiques des clusters:")
                st.write("Nombre de points de données dans chaque cluster:", np.bincount(clusters))
                st.write("Centres des clusters:", kmeans.cluster_centers_)

            elif clustering_algo == 'DBSCAN':
                eps = st.number_input("Valeur de epsilon", min_value=0.1, value=0.5)
                min_samples = st.number_input("Nombre minimum d'échantillons", min_value=1, value=5)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                imputer = SimpleImputer(strategy='mean')
                df_imputed = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[int, float])), columns=df.select_dtypes(include=[int, float]).columns)
                clusters = dbscan.fit_predict(df_imputed)
                df['Cluster'] = clusters

                fig = px.scatter_matrix(df, dimensions=df_imputed.columns, color='Cluster', title="Visualisation des clusters DBSCAN")
                st.plotly_chart(fig)

                unique, counts = np.unique(clusters, return_counts=True)
                st.write("Statistiques des clusters:")
                st.write("Nombre de points de données dans chaque cluster:", dict(zip(unique, counts)))
                st.write("Densité des clusters:")
                for cluster in unique:
                    if cluster != -1:
                        st.write(f"Cluster {cluster}: {counts[cluster] / len(clusters[clusters == cluster])}")

if __name__ == "__main__":
    main()
