# MRI Classification

La classification de maladie neurologique a pour but d’améliorer la prise en charge des patients et de leur donner le traitement le plus approprié. Pour cela, un algorithme de deep learning a été développé pour classer 5 maladies neurologiques grâce à des IRM axial T2.

## Installation

```bash
pip install glob
pip install numpy 
pip install pandas
pip install Pillow
pip install argparse
pip install matplotlib
pip install scikit-learn
Pip install opencv-python
pip install albumentations
pip install torch, torchvision, torchmetrics
```

## Exploration des données 

Description des données :
1. Général :

Données : 1296 IRM au total de type T2 axial
Classes : 5 maladies representées 
(EO: “Early Onset Neuroflux disorder”
IO: “Intermediate onset Neuroflux disorder”
LO : ”late Onset Neuroflux disorder”
PTE: ”Neuroflux disorder with polyglutamine tract expansion”
IPTE : ”Neuroflux disorder with intermediate polyglutamine tract expansion”)

Dans le jeu de données, il y a un déséquilibre des données à prétraiter avant d'utiliser un algorithme d'intelligence artificielle.

|Classes|0 (PTE)|1 (IO)|2 (IPTE)|3 (EO)|4 (LO)||------||:------:||:------:||:------:||:------:||------:||Avant équilibrage|580|72|171|233|240||------||------||------||------||------||------||Après équilibrage|171|72|171|171|171|

Pour finir, un resizing et une normalisation des données est faite.

## Entrainement de modèle de prédiction

Les algorithmes utilisés sont :

- VGG16 avec des poids pré-entrainés
- CNN

Executer la commande python main_MRICLassification.py —-MRIfolder_neuroflux <neuroflux MRI folder> —-model_name <"VGG16 or "CNN">

## Analyse statistique des performances
VGG16 :
|ACC | Precision | Recall | Specificity | F1 Score |
|------||------||------||------||------|
|71,4%||12,0%||15,0%||81,0%||42,9%|
  
 CNN :
| ACC | Precision | Recall | Specificity | F1 Score |
|------||------||------||------||------|
|100%||2,9%||20,0%||80,0%||14,3%| 
  
L'algorithme n'arrive pas à apprendre

## Dockerisation
  
 docker pull ghcr.io/atfehrana/classification/test:latest 
  

## Pistes poursuite

- Analyser les données entre elles 

- Ajout de nouvelles informations (antécédents)

- Augmenter le jeux de données (amélioration du modèle)

- Utilisation d'autres modèles d'IA

## Documentation

[vgg 16] (https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)

## Authors

- Rana Atfeh

