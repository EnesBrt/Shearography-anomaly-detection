# Détection de défauts par shearographie avec IA

Langue / Language : Français | [English](README.eng.md)

## Vue d’ensemble du projet
Ce dépôt contient une preuve de concept d’un pipeline IA pour la détection de défauts sur des images de shearographie à l’aide d’Ultralytics YOLO et du dataset SADD. Le projet se concentre sur la détection et la localisation de zones défectueuses, et inclut à la fois les éléments d’entraînement et une application Streamlit locale pour l’inférence.

Le travail suit une approche en deux étapes :
- Niveau 1 : détection binaire entraînée uniquement sur des images défectueuses
- Niveau 2 : détection binaire avec ajout d’images saines négatives (`good_clean` et `good_stripes`) afin de réduire les faux positifs

## Problématique
L’inspection manuelle d’images de shearographie peut être longue et subjective, en particulier lorsque des motifs de déformation sains ressemblent visuellement à de vrais défauts. L’objectif de ce projet est de construire un assistant IA pratique qui :
- détecte la présence d’un défaut
- localise les zones suspectes avec des boîtes englobantes
- aide à réduire les faux positifs sur des échantillons sains mais visuellement difficiles
- peut être démontré au travers d’une application locale légère

## Dataset
Ce projet s’appuie sur le dataset SADD (Shearographic Anomaly Detection Dataset).

Catégories pertinentes utilisées dans ce dépôt :
- `faulty` : échantillons défectueux avec annotations des régions de défaut
- `good_clean` : échantillons sains sans annotations de défaut
- `good_stripes` : échantillons sains avec motifs de déformation en bandes pouvant perturber le détecteur

Configuration YOLO utilisée dans ce dépôt :
- classe unique de détection : `fault`
- `SD_YOLO/` : dataset Niveau 1
- `SD_YOLO_L2/` : dataset Niveau 2 avec ajout de négatifs sains sous forme d’images à labels vides

Dans cette formulation :
- `good_clean` et `good_stripes` ne sont pas des classes de détection distinctes
- ce sont des images négatives utilisées pour améliorer la robustesse et le comportement face aux faux positifs

## Méthodologie
Le projet utilise Ultralytics YOLO pour la détection d’objets.

Workflow global :
1. Préparer un dataset Niveau 1 à partir des seules images défectueuses
2. Entraîner et valider un détecteur binaire pour la classe `fault`
3. Exporter le modèle entraîné au format ONNX pour une inférence orientée déploiement
4. Construire un dataset Niveau 2 en mélangeant :
   - images défectueuses avec labels
   - images saines `good_clean` avec labels vides
   - images saines `good_stripes` avec labels vides
5. Réentraîner puis comparer le Niveau 2 au Niveau 1
6. Utiliser le modèle ONNX exporté dans une application locale Streamlit de démonstration

Priorités d’évaluation :
- métrique principale : `mAP50-95`
- métriques complémentaires : précision et rappel

## Résultats
Le dossier `results/` contient les principaux artefacts d’entraînement et de validation actuellement suivis dans le projet, notamment :
- `results.csv` : métriques d’entraînement et de validation par époque
- `results.png` : courbes récapitulatives d’entraînement
- `confusion_matrix.png`
- `confusion_matrix_normalized.png`
- `val_batch2_pred.jpg` : exemple de prédiction sur le lot de validation

#### Courbes d'entraînement
![Courbes d'entraînement](results/results.png)

#### Matrice de confusion
![Matrice de confusion](results/confusion_matrix.png)

#### Matrice de confusion normalisée
![Matrice de confusion normalisée](results/confusion_matrix_normalized.png)

#### Exemple de prédiction sur le jeu de validation
![Exemple de prédiction](results/val_batch2_pred.jpg)

## Inférence / déploiement
Ce dépôt inclut une application locale Streamlit :
- `app.py`

L’application :
- charge le modèle ONNX
- prend en charge des images intégrées et des uploads personnalisés
- affiche les prédictions annotées
- utilise le mode sombre par défaut
- inclut des exemples intégrés pour :
  - `fault`
  - `good_clean`
  - `good_stripes`

Explication simplifiée affichée dans l’application :
- Fault : défaut réel
- Good clean : image saine
- Good stripes : image saine avec motifs de déformation

## Structure du dépôt
Vue simplifiée du dépôt :

```text
shearography_ai/
├── README.md
├── README.fr.md
├── app.py
├── pyproject.toml
├── .streamlit/
│   └── config.toml
├── notebook/
│   └── shearo_model.ipynb
├── results/
│   ├── results.csv
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   └── val_batch2_pred.jpg
├── SD_YOLO/
│   └── data.yaml
├── SD_YOLO_L2/
│   └── data.yaml
├── detect_L2/
│   └── train/weights/
│       ├── best.pt
│       └── best.onnx
└── app_assets/
    ├── sample_index.json
    └── sample_inputs/
```

## Utilisation
### 1. Créer et activer un environnement virtuel
Exemple :

```bash
cd shearography_ai
python -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances
```bash
pip install -U pip
pip install -e .
```

Si besoin, installer directement les dépendances du projet :
```bash
pip install ultralytics streamlit onnx onnxruntime pandas
```

### 3. Lancer l'app Streamlit
```bash
streamlit run app.py
```

### 4. Lancer une prédiction avec Ultralytics

#### Prédiction sur une image avec le modèle ONNX
```bash
yolo predict task=detect model=detect_L2/train/weights/best.onnx source=app_assets/sample_inputs/fault_001.png imgsz=640 conf=0.25 save=True
```


#### Prédiction sur un dossier d'images avec le modèle ONNX
```bash
yolo predict task=detect model=detect_L2/train/weights/best.onnx source=app_assets/sample_inputs imgsz=640 conf=0.25 save=True
```

Notes :
- les exemples intégrés sont gérés via `app_assets/sample_index.json`
- les sorties de prédiction Ultralytics sont généralement enregistrées dans un dossier `runs/detect/predict*`

## Références
- SADD — Shearographic Anomaly Detection Dataset: [https://github.com/jessicaplassmann/SADD](https://github.com/jessicaplassmann/SADD)
- Ultralytics YOLO: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- Automated Annotation of Shearographic Measurements Enabling Weakly Supervised Defect Detection: [https://arxiv.org/abs/2512.06171](https://arxiv.org/abs/2512.06171)
