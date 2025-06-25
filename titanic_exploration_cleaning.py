# titanic_exploration_cleaning.py
# Author: Alwin Philipose
# Description: Data exploration and cleaning on Titanic dataset (Kaggle-style)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Plot boxplot of Fare


# Load dataset
df = pd.read_csv(r'C:\Users\Alwin\Documents\Dataset for Data Analysis\titanic\train.csv')

# Show first few rows
print("Initial Data Preview:")
print(df.head())

# Basic summary
print("\nSummary statistics:")
print(df.describe(include='all'))

# Print the categorical columns data
categorical= df.dtypes[df.dtypes== "object"].index
print(categorical)

print(df[categorical].describe())


# Data Types
print("\n Data types:")
print(df.dtypes)

print("\nDo I need all the variables ? Lets figure out then remove the junk columns that arent useful for our analysis")
# For instance Passenger ID is of no use 

del df["PassengerId"]
print("Initial Data column named PassengerID deleted:")
print(df.head())

print("\n Now TRANSFORMATION OF DATA: variable with column name Survived to two values Died or Survived for readability")

new_survived= pd.Categorical(df["Survived"])
new_survived= new_survived.rename_categories(["Died","Survived"])
df["Survived"]= new_survived 
print(df.head(5))
print(new_survived.describe())

print("\n Now TRANSFORMATION OF DATA: Lets group the cabin by Letter. Now extract the first Letter from each object")

char_Cabin= df["Cabin"].astype(str)
new_cabin= np.array([cabin[0] for cabin in char_Cabin])
new_cabin= pd.Categorical(new_cabin)
new_cabin.describe()

df["Cabin"]= new_cabin

print(df["Cabin"])






# Check for missing values
print("\nMissing values in every column:")
print(df.isnull().sum())

# Fill missing Age values with the median of Age
print(df["Age"].describe())
print("\nWhen checked the Age column, it was found that Count is fewer than the total number of records, so lets fill empty values with the MEDIAN of AGE")
median_age = df['Age'].median()
new_age= np.where(df["Age"].isnull(), median_age, df["Age"])  
df["Age"]=new_age
print(f"\nFilled missing 'Age' with median: {median_age}")
print(df["Age"])

# Fill missing Embarked values with the mode (most common)
print(df["Embarked"].describe())
missing=np.where(df["Embarked"].isnull()==True)
print(f"\n The missing Embarked values are on the following indexes of the array: {missing}")

print("\n Two values of Embarked column are missing and the count is 889 so lets fill it by the mode of all the values in Embarked column and then the count should be 891")
mode_embarked = df['Embarked'].mode()[0]
mode_embarked=np.where(df["Embarked"].isnull(), mode_embarked, df["Embarked"])
#df['Embarked'].fillna(mode_embarked, inplace=True)
print(f"\nFilled missing 'Embarked' with mode: {mode_embarked}")
df["Embarked"]=mode_embarked
print(df["Embarked"])
print(df["Embarked"].describe())

# Drop columns with too many missing values (e.g., 'Cabin')
missing_threshold = 0.5  # 50%
missing_fraction = df.isnull().mean()
columns_to_drop = missing_fraction[missing_fraction > missing_threshold].index
df.drop(columns=columns_to_drop, inplace=True)
print(f"\nDropped columns with >50% missing values: {list(columns_to_drop)}")

# Convert 'Sex' to numeric using one-hot encoding
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Feature Engineering: Create FamilySize from SibSp + Parch
df['FamilySize'] = df['SibSp'] + df['Parch']
print("\nAdded new feature 'FamilySize'.")
print(f"\n The maximum members in a family are ",
       df["FamilySize"].max())
most_family= np.where(df["FamilySize"]==max(df["FamilySize"]))
print(f"\nThe following are the families with maximum number of members\n",
      df.loc[most_family])


# Outlier Detection (simple rule): flag very high fares. First lets find those cases using plot

df["Fare"].plot(kind="box", figsize=(9, 9))
plt.title("Boxplot of Fare")
plt.ylabel("Fare")
plt.show()  # <- This actually renders the plot window!

maxfare= np.where(df["Fare"]==max(df["Fare"]))
print(f"\n The indexes of the people who paid the most are\n", 
      df.loc[maxfare])

df = df[df["Fare"] != df["Fare"].max()]

print(df["Fare"].describe())

df['Fare_outlier'] = np.where(df['Fare'] > 100, 1, 0)
print("\nFlagged outliers in Fare column.")

# Final cleaned data summary
print("\nCleaned Data Preview:")
print(df.head())

# Save cleaned dataset
df.to_csv("titanic_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'titanic_cleaned.csv'")


