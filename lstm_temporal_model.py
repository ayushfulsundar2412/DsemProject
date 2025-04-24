import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("temporal_data.csv")

# Define time steps and features
years = range(2000, 2007)
feature_names = ["MAX_SYSTOLIC", "MAX_DIASTOLIC", "MAX_BMI", "MAX_VALUE_Chol"]

# Prepare LSTM input: (samples, time_steps, features)
X_lstm = []
for _, row in df.iterrows():
    sequence = []
    for year in years:
        values = [row[f"{f}_{year}"] for f in feature_names]
        sequence.append(values)
    X_lstm.append(sequence)

X_lstm = np.array(X_lstm)
y_lstm = df["Class"].values

# Normalize features
n_samples, n_steps, n_features = X_lstm.shape
X_flat = X_lstm.reshape(-1, n_features)
scaler = StandardScaler().fit(X_flat)
X_scaled = scaler.transform(X_flat).reshape(n_samples, n_steps, n_features)

# One-hot encode labels
y_cat = to_categorical(y_lstm)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(n_steps, n_features)))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=16, verbose=1)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No CVD', 'CVD'], yticklabels=['No CVD', 'CVD'])
plt.title("LSTM Model - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred_classes))
