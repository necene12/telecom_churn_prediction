## Déploiement de modèles à travers FastApi, Docker et Kubernetes
Ce répertoire contient le code et les différentes étapes relatives à : 
 - Le nettoyage et exploration des données.
 - La contruction, choix et sauvegarde des modèles.
 - La construction de l'API avec FastApi, contenant les routes: status, prediction unitaire, prédiction par fichier, réentrainement et sauvegarde du modèle.
 - La construction de l'image et du container Docker.
 - Au déploiement de l'API et du container sur Kubernetes.
 
Les fichiers relatifs aux différentes étapes sont les suivants: 
- preprocessing.ipynb: Notebook qui permet de nettoyer et explorer les données.
- construction_sauvegarde_de_modeles.ipynb : Pour la construction et sauvegarde des modèles à partir des données sources.
- model_AdboostClassifier.pkl : fichier de sortie du modèle adboost classifier
- model_regressionlogistique.pkl : fichier de sortie du modèle de regressions linéaire
- setup.sh permet de construire l'image exécuter le container et sauvegarder l'image
- donnees_test_predictions.csv : contient les données sources qui serviront à tester le modèle sur le point de terminaison de prédiction en masse de l'api
- api_fastapi.py: le fichier de l'api
- Dockerfile: qui contient les instruction pour la création et l'exécution du container docker

Exécuté les commandes suivante à la racine du répertoire /Apps pour construire l'image docker, la sauvegarder et l'exécuter
  cd /Apps
  sh setup.sh

- Installer kubectl sur votre machine et exécuter les commandes suivantes:
  kubectl create -f churn_prediction_deployment.yml
- gestion du service pour l'exposition de l'api
  kubectl create -f churn_prediction_service.yml
  kubectl expose deploy churn_prediction_deployment --type=ClusterIP --port=8002 --target-port=8000 --name churn_prediction_service
- gestion de l'ingress pour l'exposition de l'api
  minikube addons enable ingress
  kubectl create -f churn_prediction_ingress.yml
- obtenir l'adresse ip et le port de l'ingress par la commande suivante sur le quel l'api sera exposé
  kubectl get ingress
- executer la commande suivante pour déployer l'api à l'adresse et le port de l'ingress
  ssh -i "nom du fichier contenant la clé ssh" ubuntu@IP_LOCALE_DE_LA_MACHINE -fNL 8000:IP_ingress
- Chargez l'adress suivante sur votre navigateur pour essayer les points de terminaison de l'api directement sur l'interface OpenAPI de fastApi
  http://127.0.0.1:8000/docs