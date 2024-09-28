from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle de régression linéaire
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    mileage = float(request.form['mileage'])
    age = float(request.form['age'])
    
    # Faire la prédiction
    prediction = model.predict(np.array([[mileage, age]]))
    return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

