# Cancer Data Prediction with Multiple Models
# This notebook demonstrates how to predict cancer diagnoses using various machine learning models, including Simple Linear Regression, Multiple Linear Regression, Polynomial Regression, SVR, Random Forest, Decision Tree, Logistic Regression, KNN, SVM, Kernel SVM, and Naive Bayes. We will also perform hyperparameter tuning using Grid Search to optimize model performance.
## Step 1: Import Required Libraries
# Importing necessary libraries for data handling, visualization, and machine learning models.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV # For splitting the dataset and performing hyperparameter tuning
from sklearn.preprocessing import StandardScaler, PolynomialFeatures # For data scaling and polynomial features
from sklearn import svm # Support Vector Machine model
from sklearn.ensemble import RandomForestClassifier # Random Forest model
from sklearn.linear_model import LogisticRegression, LinearRegression # Logistic and Linear Regression models
from sklearn.metrics import accuracy_score # For calculating accuracy of models
from sklearn.neighbors import KNeighborsClassifier # K-Nearest Neighbors model
from sklearn.tree import DecisionTreeClassifier # Decision Tree model
from sklearn.svm import SVR # Support Vector Regression model
from sklearn.naive_bayes import GaussianNB # Naive Bayes model
import matplotlib.pyplot as plt # For visualizing results

# ## Step 2: Load and Preprocess the Data

# %%
# Load the dataset into a pandas DataFrame
data = pd.read_csv("Cancer_Data.csv")

# %%
data.head()

# %%
data.columns

# %%
# Drop unnecessary columns ('Unnamed: 32' and 'id') that don't provide useful information for prediction
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# %%
# Convert diagnosis column into binary values: 1 for malignant (M), 0 for benign (B)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

# %%
# Separate features (x_data) and target variable (y)
x_data = data.drop(["diagnosis"], axis=1) # Features are all columns except 'diagnosis'
y = data.diagnosis.values # Target variable is 'diagnosis'

# %%
# Normalize the feature data using min-max scaling
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())

# %%
# Now x contains the normalized features, and y contains the binary target
print(x.head())
print(y[:5])

# %% [markdown]
# ##  Step 3: Split the Data into Training and Test Sets

# %%
# Split the data into 80% training and 20% testing to evaluate model performance
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# %%
# Standardize the features
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# %% [markdown]
# ## Step 4: Define Model Parameters for Grid Search

# %%
# Define model parameters for Grid Search
model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear']
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1, 5, 10]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(solver='liblinear', multi_class='auto'),
        'params': {
            'C': [1, 5, 10]
        }
    },
    'knn': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 10]
        }
    },
    'decision_tree': {
        'model': DecisionTreeClassifier(),
        'params': {
            'max_depth': [None, 5, 10]
        }
    },
    'naive_bayes': {
        'model': GaussianNB(),
        'params': {}
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params': {}
    },
    'multiple_linear': {
        'model': LinearRegression(),
        'params': {}
    },
    'polynomial_regression': {
        'model': None,  # Placeholder for polynomial regression
        'params': {}
    },
    'svr': {
        'model': SVR(),
        'params': {
            'kernel': ['linear', 'rbf'],
            'C': [1, 10]
        }
    }
}


# %% [markdown]
# ## Step 5: Perform Grid Search and Collect Scores

# %%
# Initialize an empty list to store scores
scores = []

# Loop through each model defined in the model_params dictionary
for model_name, mp in model_params.items():
    # Check if the model is Polynomial Regression
    if model_name == 'polynomial_regression':
        # Transform the feature data to include polynomial terms
        poly = PolynomialFeatures(degree=2)  # Create polynomial features of degree 2
        x_train_poly = poly.fit_transform(x_train)  # Fit and transform the training data
        x_test_poly = poly.transform(x_test)  # Transform the test data using the same polynomial features

        # Initialize and fit a Linear Regression model
        clf = LinearRegression()
        clf.fit(x_train_poly, y_train)  # Fit the model to the polynomial training data

        # Make predictions on the test data and round the results to get binary values
        y_pred_poly = clf.predict(x_test_poly)
        accuracy = accuracy_score(y_test, np.round(y_pred_poly))  # Calculate accuracy of predictions

        # Append the model name, accuracy, and empty parameters to the scores list
        scores.append({
            'model': model_name,
            'best_score': accuracy,  # Store the accuracy for Polynomial Regression
            'best_params': {}  # No parameters to report for this model
        })
        continue  # Skip to the next model in the loop

    # For other models, perform Grid Search to find the best hyperparameters
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)  # Initialize Grid Search
    clf.fit(x_train, y_train)  # Fit the model to the training data

    # Append the model name, best score, and best parameters to the scores list
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,  # Store the best accuracy found during Grid Search
        'best_params': clf.best_params_  # Store the best parameters found during Grid Search
    })


# %% [markdown]
# ## Step 6: Convert Results to DataFrame and Print

# %%
# Convert scores to a DataFrame for easy visualization
df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

# %% [markdown]
# ## Step 7: Evaluate the Best Model

# %%
# Identify the best model based on the highest score in the DataFrame
best_model_name = df.loc[df['best_score'].idxmax(), 'model']  # Get the model name with the highest score
best_model_params = df.loc[df['best_score'].idxmax(), 'best_params']  # Get the corresponding best parameters

# Train the best model with the identified best parameters
if best_model_name == 'polynomial_regression':
    # For Polynomial Regression, transform the training data again
    poly = PolynomialFeatures(degree=2)  # Create polynomial features of degree 2
    x_train_poly = poly.fit_transform(x_train)  # Fit and transform the training data
    best_model = LinearRegression()  # Initialize a Linear Regression model
    best_model.fit(x_train_poly, y_train)  # Fit the model to the polynomial training data
else:
    # For other models, update the model with the best parameters
    best_model = model_params[best_model_name]['model'].set_params(**best_model_params)  # Set the best parameters
    best_model.fit(x_train, y_train)  # Fit the model to the training data

# Predict and evaluate on the test set
if best_model_name == 'polynomial_regression':
    x_test_poly = poly.transform(x_test)  # Transform the test data to polynomial features
    y_pred_best = best_model.predict(x_test_poly)  # Make predictions using the best model
else:
    y_pred_best = best_model.predict(x_test)  # Make predictions using the best model for other cases

# Calculate the accuracy of the best model on the test set
best_accuracy = accuracy_score(y_test, np.round(y_pred_best))  # Evaluate accuracy

# Print the name of the best model and its test accuracy
print(f"Best Model: {best_model_name}, Test Accuracy: {best_accuracy}")


# %% [markdown]
# ## Step 8: Visualize Model Accuracies

# %%
# Optional: Visualize the performance of different models
plt.figure(figsize=(10, 5))
plt.bar(df['model'], df['best_score'])
plt.xlabel('Models')
plt.ylabel('Best Cross-Validation Score')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.show()


# %%



