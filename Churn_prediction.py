# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

# Step 2: Load Dataset
df = pd.read_csv("Customer_Churn.csv")
print(df.head())
print(df.info())

# Step 3: Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Split data
X = df.drop('Exited', axis=1)   # (or 'Churn' depending on dataset)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train model
model = XGBClassifier(random_state=42, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate model
print("✅ Model Evaluation Results:")
print("-----------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Churn Prediction")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 10: Feature Importance
importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    'Feature Name': feature_names,
    'Importance Score': importances
}).sort_values(by='Importance Score', ascending=False)

# Save feature importance for Power BI
feat_imp.to_csv("Feature_Importance.csv", index=False)
print("✅ Feature importance saved to 'Feature_Importance.csv'")

# Plot top 10
plt.figure(figsize=(10,5))
sns.barplot(x='Importance Score', y='Feature Name', data=feat_imp.head(10))
plt.title("Top 10 Features Influencing Churn")
plt.xlabel("Importance Score")
plt.ylabel("Feature Name")
plt.show()


# Step 11: Export predictions to CSV for Power BI visualization
df['Churn_Prediction'] = model.predict(X)
df.to_csv("Churn_Predictions.csv", index=False)
print("✅ Churn predictions saved to 'Churn_Predictions.csv'")