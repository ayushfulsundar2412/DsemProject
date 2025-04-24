
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Load data
benchmark_data = pd.read_csv("data/benchmark/data.csv")
temporal_data = pd.read_csv("data/temporal/data.csv")




X = benchmark_data.drop(columns=['GRID', 'Class'])  # Dropping GRID (string ID) and Class (target)
y = benchmark_data['Class']

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Encode y if necessary
if y.dtype != 'int':
    y = pd.factorize(y)[0]

def benchmark_models(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=5000),
        # 'SVM': SVC(probability=True)  # Temporarily disable for speed
    }

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        acc_scores = []
        auc_scores = []
        for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"  Fold {i+1}/5")
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = model.predict(X.iloc[test_idx])
            probas = model.predict_proba(X.iloc[test_idx])[:,1]
            acc_scores.append(accuracy_score(y[test_idx], preds))
            auc_scores.append(roc_auc_score(y[test_idx], probas))
        results[name] = {
            'accuracy': np.mean(acc_scores),
            'roc_auc': np.mean(auc_scores)
        }

    return results

# DNN training
def train_dnn(X, y):
    y_cat = to_categorical(y)
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(y_cat.shape[1], activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit(X, y_cat, epochs=50, validation_split=0.2, callbacks=[early_stop], verbose=0)

    final_acc = history.history['val_accuracy'][-1]
    return model, final_acc

if __name__ == "__main__":
    print("Running benchmark models...")
    benchmark_results = benchmark_models(X, y)
    for model_name, scores in benchmark_results.items():
        print(f"{model_name}: Accuracy = {scores['accuracy']:.4f}, ROC-AUC = {scores['roc_auc']:.4f}")

    print("\nTraining Deep Neural Network...")
    dnn_model, dnn_acc = train_dnn(X, y)
    print(f"DNN Validation Accuracy: {dnn_acc:.4f}")

# Save results to CSV
import csv

with open("cvd_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model", "Accuracy", "ROC_AUC"])
    for model_name, scores in benchmark_results.items():
        writer.writerow([model_name, scores['accuracy'], scores['roc_auc']])
    writer.writerow(["DNN", dnn_acc, "N/A"])

print("\n✅ Results saved to cvd_results.csv")

print("\n✅ Results saved to cvd_results.csv")

# Plotting Accuracy Comparison
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cvd_results.csv")
df.set_index("Model")[["Accuracy"]].plot(kind="bar", title="Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.85, 0.95)
plt.tight_layout()
plt.show()
