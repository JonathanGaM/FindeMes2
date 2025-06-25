from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo PM2.5
try:
    modelo = joblib.load('modelo_pm25.pkl')
    app.logger.debug('Modelo PM2.5 cargado correctamente.')
    app.logger.debug(f'Características: {modelo["caracteristicas"]}')
    app.logger.debug(f'Tipo de modelo: {modelo["tipo"]}')
    app.logger.debug(f'R²: {modelo["r2"]:.4f}')
except Exception as e:
    app.logger.error(f'Error cargando modelo: {str(e)}')
    modelo = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if modelo is None:
            return jsonify({'error': 'Modelo no disponible'}), 500
            
        # Obtener datos del formulario para las 5 características
        pm10 = float(request.form['pm10'])
        co = float(request.form['co'])
        so2 = float(request.form['so2'])
        no2 = float(request.form['no2'])
        dewp = float(request.form['dewp'])
        
        app.logger.debug(f'Datos recibidos: PM10={pm10}, CO={co}, SO2={so2}, NO2={no2}, DEWP={dewp}')
        
        # Crear DataFrame con las características en el orden correcto
        datos = pd.DataFrame([[pm10, co, so2, no2, dewp]], 
                            columns=modelo['caracteristicas'])
        
        # Escalar datos
        datos_escalados = modelo['scaler'].transform(datos)
        
        # Realizar predicción
        prediccion = modelo['modelo'].predict(datos_escalados)[0]
        
        # Clasificar calidad del aire
        if prediccion <= 12:
            nivel = "Buena"
            color = "success"
        elif prediccion <= 35:
            nivel = "Moderada"
            color = "warning"
        elif prediccion <= 55:
            nivel = "Dañina para grupos sensibles"
            color = "danger"
        else:
            nivel = "Dañina para todos"
            color = "danger"
        
        app.logger.debug(f'Predicción realizada: {prediccion:.1f} μg/m³ - {nivel}')
        
        resultado = {
            'pm25': round(prediccion, 1),
            'nivel': nivel,
            'color': color,
            'modelo': modelo['tipo'],
            'r2': round(modelo['r2'], 4)
        }
        
        return jsonify(resultado)
        
    except ValueError as e:
        app.logger.error(f'Error de validación: {str(e)}')
        return jsonify({'error': 'Datos inválidos. Verifique los valores ingresados.'}), 400
    except Exception as e:
        app.logger.error(f'Error en predicción: {str(e)}')
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'model_loaded': modelo is not None,
        'model_info': {
            'tipo': modelo['tipo'] if modelo else None,
            'r2': modelo['r2'] if modelo else None,
            'caracteristicas': modelo['caracteristicas'] if modelo else None
        }
    })

@app.route('/examples')
def examples():
    """Endpoint para obtener ejemplos de uso"""
    return jsonify({
        'dia_limpio': {
            'pm10': 30, 'co': 1.2, 'so2': 10, 'no2': 25, 'dewp': 5,
            'descripcion': 'Día con buena calidad del aire'
        },
        'dia_contaminado': {
            'pm10': 150, 'co': 4.0, 'so2': 50, 'no2': 80, 'dewp': -10,
            'descripcion': 'Día con alta contaminación'
        }
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)