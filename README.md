# QuantBIOI-project-code

This repo contains multiple data files in csv format. Multiple scripts use these so they shouldn't be changed or deleted. 

Python Files:

MonteCarloTesting.py: 

	Run this file in the command line by navigating to its enclosing folder and running the command: python3 MonteCarloTesting.py

	This runs a user indicated number of iterations of monte carlo simulation and subsequent classification across multiple algorithms. Upon completion it displays some relevant statistics on the runs including the mean accuracy for original training data classifications, mean accuracy for monte carlo training data, shapiro-wilks normality tests for the distribution of accuracies and depending on the normalities, either ANOVA or kruskal wallis tests across those distributions. 

	NOTE: Some of the original data runs have no element of randomization so the shapiro wilks normality test will not be relevant for those. All accuracies for those models would be identical. 

