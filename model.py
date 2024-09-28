import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Charger les données
data = pd.read_csv('./data/carprices.csv')

# Afficher les noms des colonnes pour le débogage
print(data.columns)

# Supprimer les espaces des noms de colonnes, si nécessaire
data.columns = data.columns.str.strip()

# Préparer les données avec les bons noms de colonnes
X = data[['Mileage', 'Age(yrs)']]  # Utilisez les noms exacts
y = data['Sell Price($)']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
