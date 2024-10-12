# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.


## Etape 0 : 
Créer un environnement virtuel : 
```
virtualenv env
source env/bin/activate
pip install dvc
```

Créer un fichier requirements.txt à la racine du dossier associé au repo avec les dépendances suivantes : 
```
# local package
-e .

# external requirements
dvc[s3]
click
Sphinx
coverage
#awscli>=1.29.0
flake8
pandas
#logging
numpy 
pathlib 
scikit-learn
imbalanced-learn
joblib
#sys
#json
```

Créer un fichier setup.py pour permettre l'installation des dépendances du fichier requirements.txt :
```
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='This project is a starting Pack for MLOps project. It s not perfect so feel free to make some modifications on it.',
    author='DataScientest',
    license='MIT',
)
```

Installer les dépendances du fichier requirements.txt :
```
pip install -r ./requirements.txt
```


## Etape 1 : création des scripts

### Import des données 
On réutilise les fichiers check_structure.py et import_raw_data.py.  
On modifie dans ce dernier le nom du dossier pré-existant (./data/raw_data), des fichiers (raw.csv) et l'adresse du storage s3.  
Cela importe normalement le fichier "raw.csv", sur lequel on va travailler, dans le dossier existant "./data/raw_data" depuis le AWS S3.

### Split des données
On réutilise le fichier make_dataset.py.  
On conserve l'opération de train_test_split et l'écriture des fichiers dans le dossier existant "./data/processed_data".

### Normalisation des données 
Les données sont dans des échelles très variées donc une normalisation est nécessaire.  
En sortie, ce script créera deux nouveaux datasets : (X_train_scaled, X_test_scaled) que l'on sauvegarde également dans "./data/processed_data".

### Parameters tuning
On exécute une GridSearch des meilleurs paramètres à utiliser pour la modélisation sur un modèle de regression choisi au préalable.  
À l'issu de ce script, les meilleurs paramètres sont sous forme de fichier .pkl que l'on sauvegarde dans le dossier "./models".

### Entraînement du modèle
On entraîne le modèle en utilisant les paramètres trouvés à travers la GridSearch.  
On sauvegarde le modèle entraîné dans le dossier models.

### Evaluation du modèle 
Finalement, en utilisant le modèle entraîné,  on évalue ses performances.  
À l'issu de ce script,, on a un nouveau dataset dans "./data/prediction_data" qui contient les prédictions sur le test set.  
Le dossier "./metrics" contient quant à lui un fichier scores.json avec les métriques d'évaluation de notre modèle (i.e. mse, r2, etc).

## Etape 2 : connection de votre dépôt à DagsHub
Connecter ce dépôt à votre compte DagsHub.  
Faire de DagsHub l'emplacement distant pour le suivi de la donnée sans oublier d'adapter le .gitignore.  

## Etape 3 : pipeline DVC
À l'aide des commandes DVC vues dans le cours, mettre en place une pipeline qui reproduit le workflow du modèle.  

## Etape 4 : rendu
Pour rendre l'examen sur la plateforme, envoyer un .zip contenant un .md avec son nom, prénom, adresse mail et le lien vers le dépôt DagsHub.  
Partager le dépôt avec https://dagshub.com/licence.pedago en le mettant comme collaborateur avec des droits de lecture seulement.   
Pour valider l'examen, sont attendus dans ce dépôt :  
- Les 5 scripts de preprocessing, modélisation et évaluation du modèle détaillés dans l'étape 1.
- Un dossier .dvc avec un fichier de config explicitant les informations par rapport à l'emplacement distant.
- Un fichier .pkl dans l'onglet _models_ de DagsHub avec le modèle entraîné.
- Un fichier .json dans le dossier metrics avec les métriques d'évaluation du modèle.
- Un fichier dvc.yaml avec les étapes de la pipeline DVC ainsi qu'un fichier dvc.lock avec les informations de la sauvegarde.
- L'onglet _data_ devra bien afficher les données.
