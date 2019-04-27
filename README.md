# QuantBIOI-project-code
This project was done to compare accuracy performance of different classification algorithms on breast cancer dataset. We also did feature selections and generated new dataset using monte carlo simulation.
Authors are: Carlee Bettler, Ian Zavitz, and Paul Okoro

The methods used are:
Classification Algorithm:
Logistic Regression
K Nearest Neighbor
Support Vector Machine
Random Forest

Dimensionality Reduction Technique:
Principal Component Analysis
Correlational Matrix
Recursive Feature Elimination
Random Forest Feature Selection

New Data Generation Technique:
Monte Carlo Simulation

Here is a workflow on how to replicate this project using the codes and data in this repo

1. The R script group_project.R was used to cleaned the original dataset so as to remove unwanted columns such sample ID. This same script was also used to carryout the dimensionality reduction techniques on the original dataset, and four dataset was created each emanating from each reduction technique. The files are: cleaned original file cancer_data.csv, correlation matrix filtered file cor_filt_data.csv, random forest filtered file rndf_filt_data.csv, PCA filtered file pca_data.csv, and recursive feature elimination filtered file rfe_filt_data.csv. Also the last few lines of this script was used to plot a bar graph of the accuracy perfomance.

2. The files created from the R script above was used in the python script stat437_project_code.py to run the classification algorithm and generate performance accuracy of each algorithm on each dataset.

3. The script MonteCarloTesting.py was used to generate new data set from only the random forest filtered dataset. The new monte carlo generated dataset was used to train the classifiers and tested on the original random forest filtered data. 



This repo contains multiple data files in csv format. Multiple scripts use these so they shouldn't be changed or deleted. 

Python Files:

MonteCarloTesting.py: 

Run this file in the command line by navigating to its enclosing folder and running the command: python3 MonteCarloTesting.py

This runs a user indicated number of iterations of monte carlo simulation and subsequent classification across multiple algorithms. 
Upon completion it displays some relevant statistics on the runs including the mean accuracy for original training data classifications, mean accuracy for monte carlo training data, shapiro-wilks normality tests for the distribution of accuracies and depending on the normalities, either ANOVA or kruskal wallis tests across those distributions. 

NOTE: Some of the original data runs have no element of randomization so the shapiro wilks normality test will not be relevant for those. All accuracies for those models would be identical. 

