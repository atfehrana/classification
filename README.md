# MRI Classification

La classification de phase de développement d'une maladie neurologique a pour but d’améliorer la prise en charge des patients et de leur donner le traitement le plus approprié. Pour cela, un algorithme de deep learning a été développé pour classer 5 phases d'une maladie neurologique "Neuroflux disorder" grâce à des IRM axial T2.

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

- Données : 1296 IRM au total de type T2 axial
- Classes : 5 phases representées 
(EO: “Early Onset Neuroflux disorder”
IO: “Intermediate onset Neuroflux disorder”
LO : ”late Onset Neuroflux disorder”
PTE: ”Neuroflux disorder with polyglutamine tract expansion”
IPTE : ”Neuroflux disorder with intermediate polyglutamine tract expansion”)

2. Equilibrage des données :

Dans le jeu de données, il y a un déséquilibre des données en fonction des classes à prétraiter avant d'utiliser un algorithme d'intelligence artificielle.

|   Class   |   Avant   |   Après   |
|----------:|-----------|-----------|
|   0 (PTE) |    580    |    171    |
|   1 (IO)  |     72    |     72    |
|   2 (IPTE)|    171    |    171    |
|   3 (EO)  |    233    |    171    |
|   4 (LO)  |    240    |    171    |

3. Preprocessing :

Pour finir, un resizing et une normalisation des données est faite.

## Entrainement de modèle de prédiction

Les algorithmes utilisés sont :

- VGG16 avec des poids pré-entrainés (ImageNet)
- CNN

Executer la commande python main_MRIClassification.py —-MRIfolder_neuroflux <neuroflux MRI folder path> —-model_name <"VGG16 or "CNN">

## Analyse statistique des performances
 
VGG16 :
 
|  Metrics  |   VGG16   |
|----------:|-----------|
|        ACC|   71,4%   |
|  Precision|   12,0%   |
|     Recall|   15,0%   |
|Specificity|   81,0%   |
|   F1 Score|   42,9%   |
  
 
 CNN :
 
|  Metrics  |    CNN    |
|----------:|-----------|
|        ACC|   100%    |
|  Precision|   2,9%    |
|     Recall|   20,0%   |
|Specificity|   80,0%   |
|   F1 Score|   14,3%   |
  
L'algorithme n'arrive pas à apprendre !
 

## Dockerisation
  
 docker pull ghcr.io/atfehrana/classification/test:latest 

## Pistes poursuite

- Analyser les données entre elles (Présence des modalités différentes ?)

- Ajout de nouvelles informations (antécédents)

- Augmenter le jeux de données (amélioration du modèle)

- Utilisation d'autres modèles d'IA

## Documentation

[vgg 16] (https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html)

## Authors

- Rana Atfeh

