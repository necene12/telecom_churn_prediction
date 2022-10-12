# Déploiement de modèles de machine learning à travers FastApi, Docker et Kubernetes
Ce répertoire contient le code et les différentes étapes relatives à : 
 - La préparation des données.
 - La contruction et le choix des modèles.
 - La construction de l'API avec FastApi, contenant les routes: status, prediction unitaire, prédiction par fichier et réentrainement et sauvegarde du modèle.
 - La construction de l'image et du container Docker.
 - Au déploiement de l'API et du container sur Kubernetes
 
Les fichiers relatifs aux différentes étapes sont les suivants: 
-
-
-
-
-

Exécutez la commande suivante pour lancer l'API
cd api
uvicorn api_fastapi:api --reload
