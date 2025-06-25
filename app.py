from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo (reemplaza con tu modelo)
try:
    # model = joblib.load('modelo_findemes.pkl')
    # Si no tienes modelo, usaremos uno de ejemplo
    model = None
    app.logger.debug('Modelo cargado correctamente.')
except Exception as e:
    app.logger.error(f'Error cargando modelo: {str(e)}')
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos del formulario
        # Ajusta estos campos según tu modelo
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        
        # Crear DataFrame con los datos
        data_df = pd.DataFrame([[feature1, feature2]], 
                              columns=['feature1', 'feature2'])
        
        # Realizar predicción
        if model is not None:
            prediction = model.predict(data_df)
            result = f"Predicción: {prediction[0]}"
        else:
            # Predicción de ejemplo si no hay modelo
            result = f"Predicción de ejemplo: {(feature1 + feature2) * 0.5:.2f}"
        
        app.logger.debug(f'Predicción realizada: {result}')
        
        return jsonify({'prediction': result})
        
    except Exception as e:
        app.logger.error(f'Error en predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True)