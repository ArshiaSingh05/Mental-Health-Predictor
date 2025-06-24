import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, ConfusionMatrixDisplay
)
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

df = pd.read_csv("cleaned_dataset.csv")
X = df.drop(columns=['treatment', 'Country', 'state'])
y = df['treatment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("🔍 Total missing values in each column:")
print(X.isnull().sum()[X.isnull().sum() > 0])
# Store models and their names
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True)
}
best_model = None
best_score = 0
best_model_name = ""

# Train and evaluate each model
for name, model in models.items():
    print(f"\n==== {name} ====")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC Score:", auc)

    # Save best model
    if auc > best_score:
        best_model = model
        best_score = auc
        best_model_name = name

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {name}")
    plt.show()
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# Saves the best model
joblib.dump(best_model, "mental_health_model.pkl")
print(f"\n✅ Best Model Saved: {best_model_name} (ROC-AUC = {best_score:.4f})")
