
# Medical Insurance Cost Prediction Project

## Overview

This GitHub repository contains the code and resources for my MSc Data Science final project, focused on predicting medical insurance costs using various machine learning models. The project explores the application of linear regression, quantile regression, and XGBoost, with an emphasis on feature selection, outlier removal, and model comparison. The goal is to develop a robust predictive model that can provide accurate estimates of medical insurance costs based on demographic, behavioral, and geographical factors.

## Project Structure

- **data/**: Contains the dataset(s) used in this project.
- **notebooks/**: Jupyter notebooks for data exploration, model development, and evaluation.
- **scripts/**: Python scripts for data preprocessing, model training, and evaluation.
- **results/**: Output data, including plots, tables, and model performance metrics.
- **README.md**: Documentation for the repository.
- **LICENSE**: License information for the project.

## Requirements

- Python 3.x
- Jupyter Notebook
- Required Python libraries: pandas, numpy, scikit-learn, statsmodels, xgboost, matplotlib, seaborn

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/shubhamvm/MScDataScience/tree/main/Data%20Science%20Project
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Data Format

### Input Data

The input dataset should be a CSV file containing the following columns:

- **age**: Age of the policyholder.
- **sex**: Gender of the policyholder.
- **bmi**: Body Mass Index.
- **children**: Number of dependents.
- **smoker**: Smoking status (yes/no).
- **region**: Geographical region of residence.
- **charges**: Medical insurance cost (target variable).

### Output Data

The output of the code includes:

- **Predicted Costs**: CSV file containing the predicted medical insurance costs.
- **Model Performance Metrics**: A summary of the mean squared error (MSE) and R² scores for each model.
- **Plots**: Visualizations of model performance, feature importance, and residual analysis.

## Code Summary

### Preprocessing

- **data_preprocessing**: Script for handling missing values, encoding categorical variables, scaling features, and splitting the data into training and testing sets.
- **outlier_removal**: Script for identifying and removing high leverage points and outliers.

### Model Training

- **linear_regression**: Script for training and evaluating a linear regression model.
- **quantile_regression**: Script for training and evaluating quantile regression models.
- **xgboost_model**: Script for training and evaluating XGBoost models.

### Model Evaluation

- **evaluation_metrics**: Script for calculating and displaying MSE and R² scores for each model.
- **visualization**: Script for generating plots, including actual vs. predicted costs, residuals, and feature importance.

## Parameters and Hyperparameters

- **Linear Regression**: No hyperparameters (simple linear regression).
- **Quantile Regression**: Quantiles (0.25, 0.5, 0.75).
- **XGBoost**:
  - `learning_rate`: Controls the step size during gradient boosting.
  - `n_estimators`: Number of boosting rounds.
  - `max_depth`: Maximum depth of each tree.
  - `subsample`: Fraction of samples used for fitting individual trees.

## Issues and Challenges

During the development of this project, several challenges were encountered:

- **Data Imbalance**: The dataset had an imbalanced distribution of medical costs, with a higher concentration of lower costs.
- **Outlier Impact**: High leverage points and outliers significantly affected model performance, necessitating their careful identification and removal.
- **Feature Selection**: Selecting the most relevant features was crucial for improving model accuracy and interpretability.
- **Hyperparameter Tuning**: Finding the optimal hyperparameters for XGBoost required extensive experimentation.

## Conclusion

This project demonstrates the effectiveness of various regression techniques in predicting medical insurance costs. The repository provides the necessary code and documentation to replicate the analysis and further explore the application of these models to similar datasets.

## Contact

For any questions or issues, please feel free to contact me via shubhamverma321@gmail.com or open an issue on this repository.
