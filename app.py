from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Définir la constante
constante = 1.0

# Classe de transformation personnalisée
class CustomTransform(BaseEstimator, TransformerMixin):
    def __init__(self, constante):
        self.constante = constante

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Ajout d'une constante à la colonne de Solde_Compte
        X['Solde_Compte_shefted'] = X['Solde_Compte'] + self.constante

        # Application d'une transformation log aux variables ayant une distribution de puissance
        for col in ['Solde_Compte_shefted', 'Durée_Appel', 'Nombre_Contacts_Campagne', 'Jours_Dernier_Contact', 'Nombre_Contacts_Précédents']:
            X[f'{col}_log'] = np.log1p(X[col].astype(float))

        # Suppression des colonnes ayant subi une transformation et devenant inutiles
        X = X.drop(columns=['Solde_Compte_shefted', 'Durée_Appel', 'Nombre_Contacts_Campagne', 'Jours_Dernier_Contact', 'Nombre_Contacts_Précédents'])

        return X

# Charger le pipeline
modele_pipeline = joblib.load('modele_pipeline.pkl')

# Définir la fonction de génération de recommandations
def generate_recommendations(cluster):
    if cluster == 0:
        cluster_characteristics = 'Individus plus âgés, avec des soldes de compte plus bas, un nombre de contacts de campagne plus élevé. Majoritairement des personnes mariées et employées dans des rôles administratifs.'
        recommendations = {
            'Offres Financières': 'Proposez des produits financiers adaptés aux personnes âgées avec des soldes de compte plus bas, tels que des plans d\'épargne ou des prêts personnels à taux réduit.',
            'Stratégies de Communication': 'Utilisez des canaux de communication traditionnels, tels que le téléphone et le courrier postal. Les messages doivent être clairs et faciles à comprendre.',
            'Campagnes Ciblées': 'Planifiez les campagnes de contact plus tôt dans le mois, en utilisant des appels téléphoniques personnalisés.'
        }
    elif cluster == 1:
        cluster_characteristics = 'Groupe d\'ndividus avec une éducation tertiaire et des contacts cellulaires fréquents. Les contacts sont souvent pendant le mois de mai.'
        recommendations = {
            'Éducation et Carrière': 'Offrez des produits et services liés à l\'éducation continue et à l\'avancement de carrière, tels que des prêts étudiants ou des services de développement professionnel.',
            'Stratégies de Communication': 'Utilisez des canaux numériques, tels que les SMS, les applications mobiles et les emails. Les messages doivent être informatifs et axés sur le développement personnel.',
            'Campagnes Ciblées': 'Planifiez des campagnes intensives en mai, offrant des promotions spéciales ou des réductions pour les services éducatifs.'
        }
    elif cluster == 2:
        cluster_characteristics = 'Un groupe avec une grande variation d\'âge, de soldes de compte variés avec des contacts plus tôt dans le mois.'
        recommendations = {
            'Offres Diversifiées': 'Proposez une gamme variée de produits financiers et de services pour répondre aux besoins diversifiés de ce groupe, allant des produits d\'épargne à des investissements.',
            'Stratégies de Communication': 'Utilisez une combinaison de canaux de communication traditionnels et numériques. Les messages doivent être personnalisés en fonction des besoins spécifiques.',
            'Campagnes Ciblées': 'Planifiez les campagnes de contact plus tôt dans le mois, en utilisant une approche multicanal pour maximiser la portée.'
        }
    elif cluster == 3:
        cluster_characteristics = 'Groupe des jeunes avec une éducation secondaire des contacts plus tard dans le mois et un taux de succès plus élevé dans les campagnes précédentes.'
        recommendations = {
            'Produits pour Jeunes': 'Offrez des produits et services adaptés aux jeunes adultes, tels que des comptes d\'épargne à haut rendement, des prêts étudiants et des conseils financiers pour les jeunes travailleurs.',
            'Stratégies de Communication': 'Utilisez principalement des canaux numériques, tels que les réseaux sociaux, les applications mobiles et les emails. Les messages doivent être dynamiques et interactifs.',
            'Campagnes Ciblées': 'Planifiez les campagnes de contact plus tard dans le mois, en utilisant des offres spéciales et des promotions pour attirer l\'attention des jeunes adultes.'
        }
    return cluster_characteristics, recommendations

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json 
    input_data = pd.DataFrame(data, index=[0])
    
    # Prédiction du cluster
    predicted_cluster = modele_pipeline.predict(input_data)[0]
    
    # Convertir en types natifs de Python
    predicted_cluster = int(predicted_cluster)
    
    # Génération des recommandations
    cluster_characteristics, recommendations = generate_recommendations(predicted_cluster)
    
    return jsonify({
        'cluster': predicted_cluster,
        'cluster_characteristics': cluster_characteristics,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
