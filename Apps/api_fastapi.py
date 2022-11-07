#Import des packages
import pandas as pd
import numpy as np
from fastapi import FastAPI,File, UploadFile
from pydantic import BaseModel, parse_obj_as
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from fastapi.responses import HTMLResponse
from typing import List
from io import BytesIO

api=FastAPI(
    title="API pour l'exécution du modèle prédictif",
    description="Sélection d'un modèle et obtention des prédictions à travers FastAPI",
    version="3.0"
)

# Route de bienvenu
@api.get('/message/',name='Bienvenue')
def get_index():
    """Affichage du message de bienvenue
    """
    return {'message':'Cette API vous permettra d\'obtenir des prédictions de churn à partir des données d\'entrée ou à partir d\'un csv'}

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

### Route de prédiction unitaires en fonctions des valeurs saisis par l'utilisateur
@api.post('/prediction_unitaire/', name='Retourne la prédiction à partir de valeurs saisies manuellement par l\'utilisateur')
def prediction_unitaire(var: VariablesExplicatives):
    """_summary_
    Args:
        var (VariablesExplicatives): _description_
    Returns:
        _type_: _description_
    """
    loaded_model = pickle.load(open("model_regressionlogistique.pkl", "rb"))
    valeurs =  [{
        'gender' : var.gender,
        'SeniorCitizen' : var.SeniorCitizen,
        'Partner' : var.Partner,
        'Dependents' : var.Dependents,
        'PhoneService' : var.PhoneService,
        'MultipleLines' : var.MultipleLines,
        'OnlineSecurity' : var.OnlineSecurity,
        'Contract' : var.Contract,
        'PaperlessBilling' : var.PaperlessBilling,
        'PaymentMethod' : var.PaymentMethod,
        'StreamingMovies' : var.StreamingMovies,
        'StreamingTV' : var.StreamingTV,
        'TechSupport' : var.TechSupport,
        'DeviceProtection' : var.DeviceProtection,
        'InternetService' : var.InternetService,
        'OnlineBackup' : var.OnlineBackup,
        'tenure' : var.tenure,
        'MonthlyCharges' : var.MonthlyCharges,
        'TotalCharges' : var.TotalCharges
    }]

    df = pd.DataFrame(valeurs)
    xpred=loaded_model.predict(df)
    return xpred.tolist()[0]
	
### Route pour  import de fichier et prédiction en masse
@api.post('/import_fichier_prediction/', name='Retourne les prédictions à partir du fichier csv sélectionné par l\'utilisateur')
def upload(file: UploadFile):
    """A partir de votre répertoire local,
    sélectionner un fichier csv contant les données des 19 variables explicatives
    pour la génération des prédictions"""
    loaded_model = pickle.load(open("model_regressionlogistique.pkl", "rb"))
    contents = file.file.read()
    buffer = BytesIO(contents)
    df = pd.read_csv(buffer, sep=';')
    buffer.close()
    file.file.close()
    df_dict = df.to_dict(orient='records')
    xpred = pd.DataFrame.from_records(df_dict)
    pred = loaded_model.predict(xpred)
    return pred.tolist()

### Route de réentrainement et sauvegrade du modèle
@api.post('/reentrainement_du_modele/',name='Entrainement du modèle sur la base des données initiales pour mise à jour')
def reentrainement():
    """entraine de sauvegarde nouveau le modèle
    """
    df = pd.read_csv("https://assets-datascientest.s3-eu-west-1.amazonaws.com/de/total/churn.csv")
    df = df.reset_index()
    df = df.replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype('float64')
    df.TotalCharges = df.TotalCharges.fillna(0)
    df['SeniorCitizen'].replace({0:'No',1:'Yes'}, inplace=True)
    df['Churn'].replace({'No':0,'Yes':1}, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
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
    colonnes_numeriques =['tenure',
                          'MonthlyCharges',
                          'TotalCharges']
    colonnes_a_supprimer='customerID'
    colonne_cible = 'Churn'
    df = df.drop('index',axis='columns')
    X = df.drop([colonne_cible, colonnes_a_supprimer], axis='columns')
    y = df[colonne_cible]

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

    pipeline_reglog = Pipeline(steps = [
                   ('preprocessor', preprocessor),
                   ('Oversampling', SMOTE()),
                   ('regressor',LogisticRegression())
               ])
    model_reglog = pipeline_reglog.fit(X, y)
    # Sauvegarde du modèle de regression logistique
    pickle.dump(model_reglog, open('model_regressionlogistique.pkl','wb'))

    # Pipeline adboost
    pipeline_adboost = Pipeline(steps = [
                   ('preprocessor', preprocessor),
                   ('Oversampling', SMOTE()),
                   ('regressor',AdaBoostClassifier())
               ])
    # Construction du modèle AdboostClassifier
    model_adboost = pipeline_adboost.fit(X,y)
    # Sauvegarde du modèle AdboostClassifier
    pickle.dump(model_adboost, open('model_AdboostClassifier.pkl', 'wb'))
    return ("Modèles sauvegardés dans le répertoire back office de l\'API")