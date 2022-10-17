#Import des packages
from os import sep
import pandas as pd
import numpy as np
from fastapi import Depends, FastAPI
from typing import List, Optional
from pydantic import BaseModel
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

api = FastAPI(
    title="API pour l'exécution du modèle prédictif",
    description="Sélection d'un modèle et obtention des prédictions à travers FastAPI",
    version="2.0"
)

# Route de bienvenu
@api.get('/',name='Bienvenue')
def get_index():
    """Affichage du message de bienvenue
    """
    return {'message':'Cette API vous permettra d\'obtenir des prédictions de churn à partir des données d\'entrée de manière unitaire ou en masse par un fichier csv'}

# route de statut de l'api
@api.get("/status", response_model=int, description="Returns 1 if API is healthy.")
async def status():
    return 1

### Préparation des colonnes pour le pipeline d'encodage et de prédiction

# Colonnes à supprimer suite au l'étape de préparation des données
colonnes_a_supprimer=['customerID']
# colonnes catégorielles
colonnes_categorielles = ['gender',
                          'SeniorCitizen',
                          'Partner',
                          'Dependents',
                          'PhoneService',
                          'MultipleLines',
                          'OnlineSecurity',
                          'Contract',
                          'PaperlessBilling',
                          'PaymentMethod',
                          'StreamingMovies',
                          'StreamingTV',
                          'TechSupport',
                          'DeviceProtection',
                          'InternetService',
                          'OnlineBackup']
# Variables numériques
colonnes_numeriques =['tenure',
                      'MonthlyCharges',
                      'TotalCharges']

# Variable cible
colonne_cible = ['Churn']

# Typage des variables explicatives 
class VariablesExplicatives(BaseModel):
    """Classe pour définition de la liste des variables sur les quels se baseront le modèle prédictif
    """
    gender:str
    SeniorCitizen:str
    Partner:str
    Dependents:str
    PhoneService:str
    MultipleLines:str
    OnlineSecurity:str
    Contract:str
    PaperlessBilling:str
    PaymentMethod:str
    StreamingMovies:str
    StreamingTV:str
    TechSupport:str
    DeviceProtection:str
    InternetService:str
    OnlineBackup:str
    tenure:int
    MonthlyCharges:float
    TotalCharges:float

# Construction du pipelines de tranformation des données:
numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean')),
       ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant')),
       ('onehot',OneHotEncoder())
])

preprocessor = ColumnTransformer(
   transformers=[
    ('numeric', numeric_transformer, colonnes_numeriques),
    ('categorical', categorical_transformer, colonnes_categorielles)
]) 

### Route de prédiction unitaires en fonctions des valeurs saisis par l'utilisateur
@api.post('/prediction_unitaire', name='Retourne les prédictions à partir de valeurs de variables saisie manuellement par l\'utilisateur')
def prediction_unitaire(variables_explicatives: VariablesExplicatives):
    """_summary_
    Args:
        variables_explicatives (VariablesExplicatives): _description_
    Returns:
        _type_: _description_
    """
    # Définition par l'utilisateur du modèle à considérer pour les prédictions
    # Import des modèles entrainés:
    # Import du modèle de régression logistique sauvegardé
    reg_log = joblib.load(open('model_regressionlogistique.pkl','rb'))
    # Import du modèle AdboostClassifier sauvegardé
    adboostclass = joblib.load(open('model_AdboostClassifier.pkl', 'rb')) 
    modele_choisi = input('Définir le modèle à utiliser = reg_log ou adboostclass?')
    # Pipeline de prédiction sur la base des données saisie par l'utilisateur
    pipeline_prediction = Pipeline(steps = [
               ('preprocessor', preprocessor),
               ('Oversampling', SMOTE()),
               ('regressor',modele_choisi)
           ])
    return [pipeline_prediction.predict(variables_explicatives)]


### Route de rédiction en masse à partir de fichier csv
@api.post('/prediction_en_masse',name='Prédiction sur la base du modèle choisi et à partir des variables explicatives d\'un fichier csv')
def prediction_en_masse():
    """Affichage des prédictions sur la base des fichier csv contenant les variables explicatives
    """
    # construction du dataframe des variable cibles d'entrée
    df = pd.read_csv(input('renseigner le nom du fichier suivi de l\'extention .csv du répertoire courant:'), sep=";")
    # suppression des variables qu'on utilisera pas pour prédire la variables cible
    X = df.drop(colonnes_a_supprimer, axis='columns')
    # Import des modèles entrainés:
    # Import du modèle de régression logistique sauvegardé
    reg_log = joblib.load(open('model_regressionlogistique.pkl','rb'))
    # Import du modèle AdboostClassifier sauvegardé
    adboostclass = joblib.load(open('model_AdboostClassifier.pkl', 'rb'))
    # Pipeline de prédiction sur la base des données saisie par l'utilisateur
    modele_choisi = input('Définir le modèle à utiliser = reg_log ou adboostclass?') 
    pipeline_prediction = Pipeline(steps = [
               ('preprocessor', preprocessor),
               ('Oversampling', SMOTE()),
               ('regressor', modele_choisi)
           ])
    return [X.append(pipeline_prediction.predict(X))]
    
### Route de réentrainement et sauvegrade du modèle

@api.post('/reentrainement_du_modele',name='entrainement du modèle sur la base des données initiale pour assurer la bonne performance du modèle')
def reentrainement():
    """entraine de sauvegarde nouveau le modèle
    """
    # Chargement des données
    dfsource = pd.read_csv("https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/churn.csv")
    dfsource = dfsource.reset_index()
    dfsource = dfsource.replace(' ', np.nan)
    dfsource['TotalCharges'] = dfsource['TotalCharges'].astype('float64')
    dfsource.TotalCharges = dfsource.TotalCharges.fillna(0)
    dfsource['SeniorCitizen'].replace({0:'No',1:'Yes'}, inplace=True)
    dfsource['Churn'].replace({'No':0,'Yes':1}, inplace=True)
    dfsource['TotalCharges'] = pd.to_numeric(dfsource['TotalCharges'])
    dfsource = dfsource.drop(colonnes_a_supprimer,axis='columns')
    # Pipeline de construction du modèle et sauvegarde
    X = dfsource.drop(colonne_cible, axis='columns')
    y = dfsource[colonne_cible]
    # Transformation des données et construction du pipeline du modèle de regressionlogistique
    pipeline_reglog = Pipeline(steps = [
                ('preprocessor', preprocessor),
                ('Oversampling', SMOTE()),
                ('regressor',LogisticRegression())
            ])
    # Construction du modèle de regression logistique
    model_reglog = pipeline_reglog.fit(X, y)
    # Pipeline adboost
    pipeline_adboost = Pipeline(steps = [
                ('preprocessor', preprocessor),
                ('Oversampling', SMOTE()),
                ('regressor',AdaBoostClassifier())
            ])
    # Construction du modèle AdboostClassifier
    model_adboost = pipeline_adboost.fit(X,y)
    joblib.dump(model_reglog, './model_regressionlogistique.pkl')
    joblib.dump(model_adboost, './model_AdboostClassifier.pkl')
    print("Les deux modèles Regression logistique et Classifieur Adboost ont été créés dans le même répertoire que celui de l'API")
