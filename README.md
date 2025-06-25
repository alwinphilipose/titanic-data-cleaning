# titanic-data-cleaning
Data cleaning and feature engineering on the Titanic dataset using Python, Pandas, and NumPy.

# Titanic Data Cleaning Project

## 🚢 Overview

This project explores and cleans the Titanic dataset from Kaggle using Python. It focuses on handling missing values, feature engineering, outlier removal, and transforming variables for further analysis or machine learning tasks.

## 📊 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib

## 📁 Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- File: `train.csv`

## 🧹 Data Cleaning & Feature Engineering Steps

- Removed irrelevant columns (`PassengerId`)
- Converted `Survived` to readable categories (`Died`, `Survived`)
- Extracted cabin letter (e.g., `C`, `E`) from full `Cabin` code
- Filled missing `Age` with median
- Filled missing `Embarked` with mode
- Dropped columns with >50% missing values
- One-hot encoded `Sex`
- Created a new feature `FamilySize`
- Detected and removed fare outliers

## 📊 Visuals

Includes a boxplot of Fare to visualize outliers.

## 🧠 Skills Demonstrated

- Data preprocessing
- Handling missing data
- Categorical transformation
- Outlier detection and feature creation
- Data visualization

## 📎 Output

Final cleaned dataset saved as `titanic_cleaned.csv`.

## 🧑‍💻 Author

Alwin Philipose  
[LinkedIn](https:www.linkedin.com/in/alwin-philipose)  
