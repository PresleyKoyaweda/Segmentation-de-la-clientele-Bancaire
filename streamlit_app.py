import streamlit as st
import requests
import pandas as pd

# Définir l'URL de l'API Flask
API_URL = "https://polar-fjord-54140-d8d1090c941e.herokuapp.com/predict"

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


# Saisir les caractéristiques de l'individu
age = st.number_input('Âge', min_value=18, max_value=100, value=40)
profession = st.selectbox('Profession', ['admin.', 'technician', 'management', 'blue-collar', 'entrepreneur', 'services', 'self-employed', 'retired', 'student', 'unemployed', 'housemaid', 'unknown'])
etat_civil = st.selectbox('État Civil', ['single', 'married', 'divorced'])
niveau_education = st.selectbox('Niveau d\'Éducation', ['primary', 'secondary', 'tertiary', 'unknown'])
avoir_credit_defaillant = st.selectbox('Avoir Crédit Défaillant', ['yes', 'no'])
solde_compte = st.number_input('Solde Compte', min_value=-10000.0, max_value=100000.0, value=0.0)
avoir_pret_logement = st.selectbox('Avoir Prêt Logement', ['yes', 'no'])
avoir_pret_personnel = st.selectbox('Avoir Prêt Personnel', ['yes', 'no'])
type_contact = st.selectbox('Type de Contact', ['cellular', 'telephone', 'unknown'])
jour_contact = st.number_input('Jour de Contact', min_value=1, max_value=31, value=1)
mois_contact = st.selectbox('Mois de Contact', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duree_appel = st.number_input('Durée Appel', min_value=0.0, max_value=1000.0, value=0.0)
nombre_contacts_campagne = st.number_input('Nombre de Contacts Campagne', min_value=0.0, max_value=100.0, value=0.0)
jours_dernier_contact = st.number_input('Jours Dernier Contact', min_value=0.0, max_value=1000.0, value=0.0)
nombre_contacts_precedents = st.number_input('Nombre de Contacts Précédents', min_value=0.0, max_value=100.0, value=0.0)
resultat_campagne_precedente = st.selectbox('Résultat Campagne Précédente', ['failure', 'success', 'other'])

# Créer un dictionnaire avec les données de l'individu
data = {
    'Âge': age,
    'Profession': profession,
    'État_Civil': etat_civil,
    'Niveau_Éducation': niveau_education,
    'Avoir_Crédit_Défaillant': avoir_credit_defaillant,
    'Solde_Compte': solde_compte,
    'Avoir_Pret_Logement': avoir_pret_logement,
    'Avoir_Pret_Personnel': avoir_pret_personnel,
    'Type_Contact': type_contact,
    'Jour_Contact': jour_contact,
    'Mois_Contact': mois_contact,
    'Durée_Appel': duree_appel,
    'Nombre_Contacts_Campagne': nombre_contacts_campagne,
    'Jours_Dernier_Contact': jours_dernier_contact,
    'Nombre_Contacts_Précédents': nombre_contacts_precedents,
    'Résultat_Campagne_Précédente': resultat_campagne_precedente
}

# Envoyer les données à l'API Flask et afficher les recommandations
if st.button('Obtenir des Recommandations'):
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Cluster prédit : {result['cluster']}")
            st.write(f"Caracteristique du cluster : {result['cluster_characteristics']}")
            st.write("Recommandations :")
            for key, value in result['recommendations'].items():
                st.write(f"**{key}**: {value}")
        else:
            st.write(f"Erreur lors de la prédiction : {response.status_code}")
    except requests.exceptions.RequestException as e:
        st.write(f"Erreur de connexion à l'API : {e}")
