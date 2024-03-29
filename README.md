### Déploiement de modèle à travers FastApi, Docker et Kubernetes
-----------------

#### Ce répertoire contient les différentes étapes : 
* Nettoyage et exploration des données.
* Contruction, choix et sauvegarde du modèle.
* Construction de l'API par FastApi, avec les routes: status, prediction unitaire, prédiction par fichier, réentraînement et  sauvegarde du modèle.
* Construction de container et de l'image Docker.
* Déploiement de l'API et du container sur Kubernetes.
-----------------

#### Description des fichiers: 
-----------------
 
```bash
 Notebook qui permet de nettoyer et explorer les données
 preprocessing.ipynb
```
```bash
 Notebook la construction et sauvegarde des modèles à partir des données sources
 construction_sauvegarde_de_modeles.ipynb 
```
```bash
 fichier de sortie du modèle adboost classifier
 model_AdboostClassifier.pkl
```
```bash
 fichier de sortie du modèle de regressions linéaire
 model_regressionlogistique.pkl
```
```bash
 permet de construire l'image Docker, exécuter et sauvegarder le container Docker
 setup.sh 
```
```bash
 contient les données sources qui serviront à tester le modèle sur la route de prédiction en masse de l'api
 donnees_pour_test_route_predictions_en_masse.csv 
```
```bash
 contient un dictionnaire de données qui serviront à tester le modèle sur la route de prédiction unitaire de l'api
 donnees_pour_test_route_prediction_unitaire.txt 
```
```bash
 le code de l'api
 api_fastapi.py 
```
```bash
 contient les instructions pour la création du container docker
 Dockerfile
```
-----------------

#### a- Installer les dépendances
-----------------
Installer les packages nécessaires au fonctionnement de l'api à travers le contenu du fichier requirements.txt
```bash
$ pip3 install -r requirements.txt
```
-----------------

#### b- Construire et sauvegarder le container Docker contenant l'API
-----------------
Exécuter le commande suivante à la racine du répertoire où les fichiers ont été clonés pour construire l'image Docker, la sauvegarder et l'exécuter
```bash
$  cd /Apps
$  sh setup.sh
  ```
-----------------

#### c- Exécuter l'API sous Kubernetes
-----------------
1. Installer kubectl sur votre machine linux et déployer le container Docker sur Kubernetes:
```bash
  sudo apt-get update && sudo apt-get install -y apt-transport-https
  curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
  echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee -a /etc/apt/sources.list.d/kubernetes.list
  sudo apt-get update
  sudo apt-get install -y kubectl
  chmod +x ./kubectl

  kubectl create -f churn_prediction_deployment.yml
```
2. Déployer de service de type clusterIP pour l'exposition de l'api
```bash
  kubectl create -f churn_prediction_service.yml
  kubectl expose deploy churn_prediction_deployment --type=ClusterIP --port=8002 --target-port=8000 --name churn_prediction_service
  ```
3. Gestion de l'ingress pour l'exposition de l'api
```bash
  minikube addons enable ingress
  kubectl create -f churn_prediction_ingress.yml
```
4. obtenir l'adresse ip et le port de l'ingress sur lequel l'api sera exposée avec la commande suivante:
```bash
  kubectl get ingress
```
5. Executer la commande suivante pour afficher l'interface graphique de l'api à l'adresse et le port de l'ingress
```bash
  ssh -i "nom du fichier contenant la clé ssh" ubuntu@IP_LOCALE_DE_LA_MACHINE -fNL 8000:IP_ingress
```
6. Charger l'adress suivante sur votre navigateur pour essayer les routes de l'api directement sur l'interface OpenAPI de fastApi
```bash
  http://127.0.0.1:8000/docs
  ```
  -----------------
