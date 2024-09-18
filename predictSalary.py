import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("employee_salary.csv")

# Separate numerical and categorical columns
numerical_cols = ['ID', 'Experience_Years', 'Age']
categorical_cols = ['Gender']

# Preprocess categorical variables using Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Define X (features) and y (target)
X = df[numerical_cols + categorical_cols]
y = df['Salary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression Section
st.title("Employee Salary Prediction")
st.write("Regression Model")

# Train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")

# Classification Section
st.title("High-Performing Employee Identification")
st.write("Classification Model")

# Create performance rating column (assuming it's missing)
df["performance_rating"] = df["Salary"] / 10000

# Preprocess data
df["high_performer"] = df["performance_rating"] >= 4

# Define X (features) and y (target) for classification
X_class = df[['Experience_Years', 'Age', 'performance_rating', 'Gender']]
y_class = df['high_performer']

# Split data into training and testing sets
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

for name, model in models.items():
    model.fit(X_train_class, y_train_class)
    y_pred_class = model.predict(X_test_class)
    accuracy = accuracy_score(y_test_class, y_pred_class)
    report = classification_report(y_test_class, y_pred_class)
    st.write(f"**{name}**")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(report)

# Prediction Form
st.title("Make a Prediction")
st.write("Enter values to predict employee salary or identify high-performing employee")

# Regression Form
st.write("Employee Salary Prediction")
salary_features = ['ID', 'Experience_Years', 'Age', 'Gender']
salary_values = [st.number_input(f"{feature}", key=f"salary_{feature}", min_value=0.0, max_value=100.0) if feature not in ['Gender'] else st.selectbox(f"{feature}", [0, 1], key=f"salary_{feature}") for feature in salary_features]
salary_pred = lr_model.predict(pd.DataFrame([salary_values], columns=salary_features))
st.write(f"Predicted Employee Salary: ${salary_pred[0]:.2f}")

# Classification Form
st.write("High-Performing Employee Identification")
performance_features = ["Experience_Years", "Age", "performance_rating", "Gender"]
performance_values = [st.number_input(f"{feature}", key=f"performance_{feature}", min_value=0.0, max_value=100.0) if feature not in ["Gender", "performance_rating"] else st.selectbox(f"{feature}", [0, 1], key=f"performance_{feature}") if feature == "Gender" else st.number_input(f"{feature}", key=f"performance_{feature}", min_value=0.0, max_value=10.0) for feature in performance_features]
performance_pred = models["Logistic Regression"].predict(pd.DataFrame([performance_values], columns=performance_features))
st.write(f"High-Performing Employee: {performance_pred[0]}")


