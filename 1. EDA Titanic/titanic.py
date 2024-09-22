import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
import missingno as msno

data = pd.read_csv("titanic.csv")

print("First 5 rows of the dataset:")
print(data.head())

print("Summary of the dataset:")
print(data.info())

print("Descriptive statistics:")
print(data.describe())

print("Missing values in the dataset:")
print(data.isnull().sum())

data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

data=data.drop(columns='Cabin', axis=1)

print("Missing values after imputation:")
print(data.isnull().sum())

plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=data)
plt.title("Survival Count")
plt.show()

# Histogram for numerical data (Age distribution)
plt.figure(figsize=(6,4))
sns.histplot(data['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Selecting only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=[np.number])

# Creating the heatmap for only numerical columns
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix (Numeric Data)")
plt.show()

print("Number of duplicate rows:", data.duplicated().sum())

# Step 6: Unique Values
print("Unique values in 'embarked' column:", data['Embarked'].unique())

print("Value counts for 'class' column:")
print(data['Pclass'].value_counts())

data['Family_size'] = data['SibSp'] + data['Parch']
print("First 5 rows with new 'family_size' feature:")
print(data.head())

# Select only numerical columns for correlation analysis
numerical_data = data.select_dtypes(include=[np.number])

# Compute correlation matrix
correlation_matrix = numerical_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

print("Data after standardization of 'age' and 'fare':")
print(data[['Age', 'Fare']].head())

z_scores = np.abs(stats.zscore(data[['Fare']]))
outliers = data[(z_scores > 3).any(axis=1)]

print(f"Number of outliers in 'fare' column: {len(outliers)}")
print("Outliers in 'fare':")
print(outliers[['Fare', 'Age', 'Pclass', 'Survived']])

print("Final dataset overview:")
print(data.info())
print("First few rows of the cleaned dataset:")
print(data.head())