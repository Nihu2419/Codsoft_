import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

# Load the datasets
train_file_path = 'E:/fraudTest.csv'
test_file_path = 'E:/fraudTest.csv'

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Automatically determine non-numeric columns and target column
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
non_numeric_cols = train_df.columns.difference(numeric_cols).tolist()
target_col = 'is_fraud'

# Fetch feature columns dynamically
feature_cols = [col for col in train_df.columns if col not in non_numeric_cols + [target_col]]

# Align columns and split data into features and target
X_train = train_df[feature_cols]
y_train = train_df[target_col]
X_test = test_df[feature_cols]
y_test = test_df[target_col]

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train a logistic regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train_scaled, y_train)

# Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logistic = logistic_model.predict(X_test_scaled)
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# ... Evaluate the models and plot ROC curves as shown in the previous responses ...


# Evaluate the models
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=1)  # Set zero_division to 1
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else None
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc, conf_matrix

acc_logistic, prec_logistic, rec_logistic, f1_logistic, roc_auc_logistic, cm_logistic = evaluate_model(y_test, y_pred_logistic)
acc_dt, prec_dt, rec_dt, f1_dt, roc_auc_dt, cm_dt = evaluate_model(y_test, y_pred_dt)
acc_rf, prec_rf, rec_rf, f1_rf, roc_auc_rf, cm_rf = evaluate_model(y_test, y_pred_rf)

# Print the evaluation metrics
print("Logistic Regression:")
print("Accuracy:", acc_logistic)
print("Precision:", prec_logistic)
print("Recall:", rec_logistic)
print("F1-score:", f1_logistic)
if roc_auc_logistic is not None:
    print("ROC AUC:", roc_auc_logistic)
print("Confusion Matrix:\n", cm_logistic)

# ... Print evaluation metrics for Decision Tree and Random Forest ...

# Plot ROC curves for all three models
y_prob_logistic = logistic_model.predict_proba(X_test_scaled)[:, 1]
fpr_logistic, tpr_logistic, _ = roc_curve(y_test, y_prob_logistic)

y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logistic, tpr_logistic, label='Logistic Regression')
plt.plot(fpr_dt, tpr_dt, label='Decision Tree')
plt.plot(fpr_rf, tpr_rf, label='Random Forest')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()