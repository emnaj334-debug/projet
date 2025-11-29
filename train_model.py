# train_model.py
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import pandas as pd

# 1) Charger le dataset Wine
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# 2) Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 4) Entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 5) Sauvegarder le modèle et le scaler
dump(model, "model.joblib")
dump(scaler, "scaler.joblib")

print("✅ Modèle et scaler créés avec succès !")
