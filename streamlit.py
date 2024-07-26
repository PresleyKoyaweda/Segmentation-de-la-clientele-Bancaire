import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Définir la classe de transformation personnalisée
class CustomTransform(BaseEstimator, TransformerMixin):
    def __init__(self, constante):
        self.constante = constante

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['Solde_Compte_shefted'] = X['Solde_Compte'] + self.constante

        for col in ['Solde_Compte_shefted', 'Durée_Appel', 'Nombre_Contacts_Campagne', 'Jours_Dernier_Contact', 'Nombre_Contacts_Précédents']:
            X[f'{col}_log'] = np.log1p(X[col].astype(float))

        X = X.drop(columns=['Solde_Compte_shefted', 'Durée_Appel', 'Nombre_Contacts_Campagne', 'Jours_Dernier_Contact', 'Nombre_Contacts_Précédents'])
        return X

# Charger le modèle
model = joblib.load('modele_pipeline.pkl')

# Définir la fonction de génération de recommandations
def generate_recommendations(cluster):
    if cluster == 0:
        cluster_characteristics = 'Individus plus âgés, avec des soldes de compte plus bas, un nombre de contacts de campagne plus élevé. Majoritairement des personnes mariées et employées dans des rôles administratifs.'
        recommendations = {
            'Offre ou Service': 'Proposez des produits financiers adaptés aux personnes âgées avec des soldes de compte plus bas, tels que des plans d\'épargne ou des prêts personnels à taux réduit.',
            'Stratégies de Communication': 'Utilisez des canaux de communication traditionnels, tels que le téléphone et le courrier postal. Les messages doivent être clairs et faciles à comprendre.',
            'Campagnes Ciblées': 'Planifiez les campagnes de contact plus tôt dans le mois, en utilisant des appels téléphoniques personnalisés.'
        }
    elif cluster == 1:
        cluster_characteristics = 'Groupe d\'ndividus avec une éducation tertiaire et des contacts cellulaires fréquents. Les contacts sont souvent pendant le mois de mai.'
        recommendations = {
            'Offre ou Service': 'Offrez des produits et services liés à l\'éducation continue et à l\'avancement de carrière, tels que des prêts étudiants ou des services de développement professionnel.',
            'Stratégies de Communication': 'Utilisez des canaux numériques, tels que les SMS, les applications mobiles et les emails. Les messages doivent être informatifs et axés sur le développement personnel.',
            'Campagnes Ciblées': 'Planifiez des campagnes intensives en mai, offrant des promotions spéciales ou des réductions pour les services éducatifs.'
        }
    elif cluster == 2:
        cluster_characteristics = 'Un groupe avec une grande variation d\'âge, de soldes de compte variés avec des contacts plus tôt dans le mois.'
        recommendations = {
            'Offre ou Service': 'Proposez une gamme variée de produits financiers et de services pour répondre aux besoins diversifiés de ce groupe, allant des produits d\'épargne à des investissements.',
            'Stratégies de Communication': 'Utilisez une combinaison de canaux de communication traditionnels et numériques. Les messages doivent être personnalisés en fonction des besoins spécifiques.',
            'Campagnes Ciblées': 'Planifiez les campagnes de contact plus tôt dans le mois, en utilisant une approche multicanal pour maximiser la portée.'
        }
    elif cluster == 3:
        cluster_characteristics = 'Groupe des jeunes avec une éducation secondaire pour la plus part avec des contacts plus tard dans le mois et un taux de succès plus élevé dans les campagnes précédentes.'
        recommendations = {
            'Offre ou Service': 'Offrez des produits et services adaptés aux jeunes adultes, tels que des comptes d\'épargne à haut rendement, des prêts étudiants et des conseils financiers pour les jeunes travailleurs.',
            'Stratégies de Communication': 'Utilisez principalement des canaux numériques, tels que les réseaux sociaux, les applications mobiles et les emails. Les messages doivent être dynamiques et interactifs.',
            'Campagnes Ciblées': 'Planifiez les campagnes de contact plus tard dans le mois, en utilisant des offres spéciales et des promotions pour attirer l\'attention des jeunes adultes.'
        }
    return cluster_characteristics, recommendations

# Fonction de prédiction
def predict_cluster(data, model):
    data_transformed = model.named_steps['custom_transform'].transform(data)
    data_preprocessed = model.named_steps['preprocessor'].transform(data_transformed)
    data_pca = model.named_steps['pca'].transform(data_preprocessed)
    cluster = model.named_steps['kmeans'].predict(data_pca)
    return cluster[0]

# Interface utilisateur avec Streamlit
st.title("Système de Recommandations de Marketing Bancaire Basées sur l'Analyse des Clusters")

# Description du projet
st.write("""
---
🔍 **Segmentation de la Clientèle avec le Dataset Bank Marketing** 🔍

Je suis Presley Koyaweda, ravi de partager avec vous mon projet de segmentation de clientèle basé sur le dataset Bank Marketing.

### 🎯 **Objectif**
Identifier des segments de clientèle distincts pour offrir des recommandations personnalisées et optimiser les stratégies de marketing.

### 🚀 **Étapes du Projet**
1. Exploration et préparation des données.
2. Transformation des données.
3. Réduction de la dimensionnalité avec PCA.
4. Segmentation des clients avec K-Means.
5. Développement d'un pipeline d'injestion, de transformation, de la reduction de dimention des données et la prediction de cluster avec des recommandations spécifiques pour chaque segment.

### 📊 **Résultats**
Quatre segments de clientèle distincts ont été identifiés, chacun avec des besoins et des comportements uniques.

### 🌟 **Conclusion**
Ce projet permet de segmenter efficacement les clients et de fournir des recommandations ciblées pour améliorer les stratégies de marketing.
         
---

Tester ce modèle en remplissant le formulaire ci-dessous pour obtenir une recommendation de la stratégie de marketing.
               
---
""")

# Entrée des caractéristiques du client
age = st.number_input('Âge', min_value=18, max_value=100, value=30)
etat_civil = st.selectbox('État Civil', ['','married', 'single', 'divorced'])
profession = st.selectbox('Profession', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'])
education = st.selectbox('Niveau Éducation', ['primary', 'secondary', 'tertiary'])
credit_defaillant = st.selectbox('Avoir Crédit Défaillant', ['yes', 'no'])
pret_logement = st.selectbox('Avoir Prêt Logement', ['yes', 'no'])
pret_personnel = st.selectbox('Avoir Prêt Personnel', ['yes', 'no'])
solde_compte = st.number_input('Solde Compte', value=0)
type_contact = st.selectbox('Type Contact', ['cellular', 'telephone'])
jour_contact = st.number_input('Jour Contact', min_value=1, max_value=31, value=15)
mois_contact = st.selectbox('Mois Contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duree_appel = st.number_input('Durée Appel', value=0)
nombre_contacts_campagne = st.number_input('Nombre Contacts Campagne', value=1)
jours_dernier_contact = st.number_input('Jours Dernier Contact', value=0)
nombre_contacts_precedents = st.number_input('Nombre Contacts Précédents', value=0)
resultat_campagne_precedente = st.selectbox('Résultat Campagne Précédente', ['failure', 'other', 'success'])

# Créer un DataFrame avec les données saisies
data = pd.DataFrame({
    'Âge': [age],
    'État_Civil': [etat_civil],
    'Profession': [profession],
    'Niveau_Éducation': [education],
    'Avoir_Crédit_Défaillant': [credit_defaillant],
    'Avoir_Pret_Logement': [pret_logement],
    'Avoir_Pret_Personnel': [pret_personnel],
    'Solde_Compte': [solde_compte],
    'Type_Contact': [type_contact],
    'Jour_Contact': [jour_contact],
    'Mois_Contact': [mois_contact],
    'Durée_Appel': [duree_appel],
    'Nombre_Contacts_Campagne': [nombre_contacts_campagne],
    'Jours_Dernier_Contact': [jours_dernier_contact],
    'Nombre_Contacts_Précédents': [nombre_contacts_precedents],
    'Résultat_Campagne_Précédente': [resultat_campagne_precedente],
    'Souscription_Dépôt_Terminé': ['no']  # Valeur fictive pour le modèle
})

# Prédire le cluster et générer des recommandations
if st.button('Prédire le Cluster'):
    cluster = predict_cluster(data, model)
    cluster_characteristics, recommendations = generate_recommendations(cluster)
    st.write(f"L'individu appartient au cluster : {cluster}")
    st.write(f"Caractéristiques du cluster : {cluster_characteristics}")
    st.write("Recommandations :")
    for key, value in recommendations.items():
        st.write(f"**{key}** : {value}")

# Pour exécuter l'application Streamlit, utilisez la commande suivante dans le terminal
# streamlit run app.py
