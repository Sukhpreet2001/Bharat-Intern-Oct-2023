
# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load and Explore Data
# Assuming you have a dataset named 'titanic_data.csv'
data = pd.read_csv('titanic_data.csv')

# Explore the dataset
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 3: Preprocess Data
# Handle missing values
data = data.dropna(subset=['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])

# Convert categorical variables to numerical
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Select features and target variable
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = data['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train the Model
# Create a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Step 5: Model Evaluation")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{classification_rep}")

# Step 6: Use the Model for Prediction
print("\nStep 6: Use the Model for Prediction")
# You can use the trained model to predict survival for new data
new_data = pd.DataFrame({'Pclass': [1], 'Sex': [1], 'Age': [25], 'SibSp': [1], 'Parch': [0], 'Fare': [100]})
prediction = model.predict(new_data)

print(f"Predicted Survival: {prediction}")
