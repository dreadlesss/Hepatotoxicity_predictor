<img src="https://github.com/dreadlesss/Hepatotoxicity_predictor/blob/master/results/TOC.jpg" width="80%">

# Instructions
This program developed a diparametric optimization method to build the hepatotoxicity (or drug-induced liver injury, DILI) prediction model. The training set consisted of 2384 DILI-labeled drugs mainly collected from the public database. Besides, 197 natural products of Polygonum multiflorum (NPPM)  were screened for the DILI potential. For detailed information, please refer to the article: [Identification of intrinsic hepatotoxic compounds in Polygonum multiflorum Thunb. using machine-learning methods](https://doi.org/10.1016/j.jep.2022.115620).

Diparametric optimization aims to construct the best prediction model by tuning the parameters of both fingerprints and machine learning (ML) models. In this paper, ECFPs, RDK fingerprints, and atom pair fingerprints were used as three examples. The parameters that can control the size of the subgraph were tuned together with the parameters of the ML algorithm. The results are shown as follows.

<img src="https://github.com/dreadlesss/Hepatotoxicity_predictor/blob/master/results/figure2.jpg" width="80%">

# Requirements 

The code is written in [Python] and mainly uses the following packages:
* [Sklearn] for model building
* [matplotlib] for plotting figures
* [rdkit] for feature extraction.

* ...
* Other codes have wrapped into [dili_predictor]

# Folders
| Folders        | Description                             |
| :----:         | :-------------------------------------: |
| data           | NPPM dataset: Polygonum_database.xlsx<br /> DILI-labeled data: reference_database.xlsx|
| dili_predictor | codes and packages used in the program  |
| results        | figures and results              |