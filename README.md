# Prédiction du risque cardiovasculaire par machine learning

Projet d'études en équipe (4e année Mathématiques Appliquées, INSA Toulouse), mené en **Python et en R**. Il répond à deux questions à partir de données de patients :

1. **Classification** : ce patient présente-t-il un risque de maladie cardiovasculaire ?
2. **Régression** : peut-on prédire son taux de **cholestérol LDL** ?

*Auteurs : Anh Hoang, Ayman Boussihmed, Felix Decker, Younès Mokhtari.*

## Données

Jeu de données synthétique de **15 000 patients**, disponible sur Kaggle (*Cardiovascular Disease Risk Prediction*). Il contient des variables démographiques (âge, sexe, poids, IMC), cliniques (tension, glycémie, cholestérol) et liées au mode de vie (tabac, alcool, activité physique, stress, sommeil, antécédents familiaux). Il n'y a ni valeur manquante ni doublon.

Les données sont découpées en apprentissage, validation et test (70 % / 15 % / 15 %), de façon stratifiée sur la variable cible.

## Démarche

**Exploration** (`exploration.ipynb`)
- Tests d'association : t de Student et χ² entre chaque variable et le risque.
- Analyse factorielle : ACP et LDA sur les variables quantitatives, ACM sur les variables qualitatives.
- Premiers constats : le tabac, les antécédents familiaux, l'activité physique et le stress sont les facteurs les plus liés au risque. Le cholestérol LDL est fortement corrélé à l'âge et au cholestérol total.

**Modélisation** (`Modelisation.ipynb` en Python, `Modelisation_R.ipynb` en R)
- Plus de 10 modèles comparés : régression logistique (sans pénalisation, Ridge, LASSO, sélection pas à pas par AIC), LDA, PLS-DA, kNN, SVM linéaire et à noyau gaussien, arbre de décision, Random Forest, Gradient Boosting / XGBoost, réseau de neurones et stacking.
- Hyperparamètres réglés par validation croisée.
- **Seuil de décision optimisé pour le F1-score** : en contexte médical, ne pas détecter un patient à risque coûte plus cher qu'une fausse alerte. Le F2-score, qui privilégie encore plus le rappel, est aussi reporté.
- Interprétation : odds ratios, importance des variables par permutation, courbes PDP et ICE.

## Résultats

### Risque cardiovasculaire (classification, jeu de test)

| Modèle | F1 | Rappel | Précision |
|---|---|---|---|
| XGBoost (R) | **0,710** | 0,788 | 0,647 |
| Régression logistique, sélection AIC (R) | 0,709 | 0,819 | 0,626 |
| Stacking des 3 meilleurs modèles (Python) | 0,696 | 0,784 | 0,626 |
| Régression logistique LASSO (Python) | 0,696 | 0,787 | 0,624 |
| SVM à noyau gaussien (Python) | 0,690 | 0,780 | 0,619 |
| kNN (Python) | 0,647 | 0,846 | 0,524 |

Les modèles linéaires font presque aussi bien que XGBoost tout en restant interprétables. La **régression logistique avec sélection AIC** est un bon compromis : 12 variables, des performances quasi identiques et des coefficients directement lisibles.

**Facteurs de risque (odds ratios) :**
- **Tabac** : environ **7 fois** plus de risque.
- **Antécédents familiaux** : environ **4 fois** plus de risque.
- **Activité physique** : risque au moins divisé par 2.
- Le stress augmente le risque, le sommeil le réduit.

### Cholestérol LDL (régression, jeu de test)

| Modèle | R² | RMSE |
|---|---|---|
| **LASSO** | **0,701** | 0,561 |
| SVR linéaire | 0,701 | 0,562 |
| Régression linéaire | 0,700 | 0,562 |
| Réseau de neurones | 0,695 | 0,567 |
| Random Forest | 0,677 | 0,584 |

Le **LASSO ne retient que 3 variables** (âge, tension systolique, cholestérol total) et obtient le meilleur score. La relation est essentiellement linéaire, et les modèles non linéaires n'apportent rien de plus.

## Reproduire

```bash
conda env create -f environment.yml
conda activate ML
```

Téléchargez le jeu de données depuis Kaggle et placez `cardiovascular_disease_data.csv` à côté des notebooks. La partie R utilise notamment `caret`, `glmnet`, `MASS`, `e1071`, `kernlab`, `randomForest`, `ranger`, `xgboost`, `keras3`, `mdatools` (PLS-DA), `pdp` et `vip`.

## Stack

Python (scikit-learn, pandas, prince, seaborn) · R (caret, glmnet, xgboost, keras3) · ACP / ACM / LDA · validation croisée · interprétabilité (odds ratios, permutation importance, PDP/ICE)
