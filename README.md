# L-MNL
Learning Multinomial Logit model (enhanced MNL with ANN)
Master thesis research project "Interpretable neural networks for a multiclass assessment of credit risk".
Based on Sifringer et al. (2020) L-MNL model.

Contents:
* Word document.

* Python files:
	- run_lending_main: main file, imports and pre-processes the data, runs a 5-Fold Stratified Cross-Validation,
			    evaluates performance measures (AUC, accuracy, F1 score, MAE, and BIC), and extracts 
			    corresponding coefficients and standard deviations for the MNL and L-MNL model.

	- models: contains the L-MNL model definition

	- grad_hess: functions for extracting betas (coefficients) and corresponding standard deviations of the L-MNL model.

* R files:
	- credit_lending: runs the Cumulative logit model, and saves the predictions in a csv file. Evaluates the Friedman test and
			  does a pairwise comparison using Wilcoxon rank sum test for the performance results of all models.

References:

Sifringer, B., Lurkin, V., & Alahi, A. (2020). Enhancing discrete choice models with representation learning. Transportation research. Part B: methodological, 140, 236-261. https://doi.org/10.1016/j.trb.2020.08.006 

## Dataset ##
George, N. (2019, April 10). All Lending Club loan data, Version 3. Retrieved February 2020 from  https://www.kaggle.com/wordsforthewise/lending-club.
