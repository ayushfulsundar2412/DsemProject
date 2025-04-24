import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# Load temporal data
df = pd.read_csv("temporal_data.csv")

# Prepare flat temporal features
years = range(2000, 2007)
features = ["MAX_SYSTOLIC", "MAX_DIASTOLIC", "MAX_BMI", "MAX_VALUE_Chol"]
all_features = [f"{f}_{y}" for y in years for f in features]

X = df[all_features]
y = df["Class"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Feature Selection with CV
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X_scaled, y)
model_fs = RandomForestClassifier(random_state=42)
fs_scores = cross_val_score(model_fs, X_selected, y, cv=5)
print(f"Feature Selection - CV Accuracy with top 20 features: {np.mean(fs_scores):.4f}")

# 2. Fairness Testing (by Gender)
df["Gender_Num"] = (df["GENDER"] == "M").astype(int)
model_fair = LogisticRegression(max_iter=5000)
model_fair.fit(X_scaled, y)
preds_fair = model_fair.predict(X_scaled)
df["Predicted"] = preds_fair

for gender in [0, 1]:
    gender_label = 'Female' if gender == 0 else 'Male'
    acc = (df[df["Gender_Num"] == gender]["Class"] == df[df["Gender_Num"] == gender]["Predicted"]).mean()
    print(f"Fairness - Accuracy for {gender_label}: {acc:.4f}")

# 3. Manual DNN CV (No scikeras)
print("\nManual DNN CV without scikeras...")

best_score = 0
best_units = None
unit_scores = []

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for units in [32, 64, 128]:
    print(f"\nEvaluating DNN with {units} units...")
    fold_scores = []

    for train_idx, test_idx in skf.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        y_train_cat = to_categorical(y_train)
        y_test_cat = to_categorical(y_test)

        model = Sequential()
        model.add(Dense(units, activation='relu', input_shape=(X.shape[1],)))
        model.add(Dense(units, activation='relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(patience=3, restore_best_weights=True)
        model.fit(X_train, y_train_cat, epochs=10, batch_size=16, verbose=0, callbacks=[es])

        _, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        fold_scores.append(acc)

    mean_acc = np.mean(fold_scores)
    print(f"  Mean CV Accuracy: {mean_acc:.4f}")
    unit_scores.append((units, mean_acc))

    if mean_acc > best_score:
        best_score = mean_acc
        best_units = units
        best_model = model

print(f"\nBest DNN Units: {best_units} with Accuracy: {best_score:.4f}")

# Save best model and logs
best_model.save("best_dnn_model.h5")
pd.DataFrame({"True": y, "Predicted": preds_fair}).to_csv("fairness_log.csv", index=False)

# 4. Mock Framingham Risk Score Calculator
def framingham_mock(age, gender, smoker, chol, sys_bp):
    score = 0
    score += age * 0.2
    score += 4 if gender == "M" else 2
    score += 3 if smoker else 0
    score += chol * 0.01
    score += sys_bp * 0.02
    return min(score, 100)

example = framingham_mock(age=55, gender="M", smoker=True, chol=210, sys_bp=145)
print(f"\nMock Framingham Risk Score (out of 100): {example:.2f}")

# === Visualization Section ===

# 1. Top 20 Feature Scores
plt.figure(figsize=(10, 5))
feature_scores = selector.scores_[selector.get_support()]
feature_names = np.array(all_features)[selector.get_support()]
sorted_idx = np.argsort(feature_scores)[::-1]

sns.barplot(x=feature_scores[sorted_idx], y=feature_names[sorted_idx], palette='viridis')
plt.title("Top 20 Feature Importance Scores")
plt.xlabel("F-score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# 2. Fairness Accuracy by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x=['Female', 'Male'], y=[
    (df[df["Gender_Num"] == 0]["Class"] == df[df["Gender_Num"] == 0]["Predicted"]).mean(),
    (df[df["Gender_Num"] == 1]["Class"] == df[df["Gender_Num"] == 1]["Predicted"]).mean()
])
plt.ylim(0, 1.05)
plt.ylabel("Accuracy")
plt.title("Fairness Accuracy by Gender")
plt.tight_layout()
plt.show()

# 3. DNN Accuracy vs Number of Units
units_list, acc_list = zip(*unit_scores)
plt.figure(figsize=(6, 4))
sns.lineplot(x=units_list, y=acc_list, marker="o")
plt.title("DNN Accuracy vs Number of Units")
plt.xlabel("Units in Hidden Layer")
plt.ylabel("CV Accuracy")
plt.ylim(0.8, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Confusion Matrix
cm = confusion_matrix(y, preds_fair)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

