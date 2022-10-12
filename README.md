## Déploiement de modèles à travers FastApi, Docker et Kubernetes
Ce répertoire contient le code et les différentes étapes relatives à : 
 - Le nettoyage et exploration des données.
 - La contruction, choix et sauvegarde des modèles.
 - La construction de l'API avec FastApi, contenant les routes: status, prediction unitaire, prédiction par fichier, réentrainement et sauvegarde du modèle.
 - La construction de l'image et du container Docker.
 - Au déploiement de l'API et du container sur Kubernetes.
 
Les fichiers relatifs aux différentes étapes sont les suivants: 
- preprocessing.ipynb: Notebook qui permet de nettoyer et explorer les données.
-
-
-
-

Exécutez la commande suivante pour lancer l'API
cd api
uvicorn api_fastapi:api --reload
