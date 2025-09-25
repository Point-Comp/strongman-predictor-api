# 3_prediction_api.py (with Debugging)
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

app = Flask(__name__)


print("Loading model and scaler...")
try:
    model = tf.keras.models.load_model('strongman_predictor.h5')
    scaler = joblib.load('data_scaler.pkl')
    athlete_profiles = pd.read_csv('athlete_profiles.csv', index_col='athlete_name')
    print("Model and artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    model, scaler, athlete_profiles = None, None, None

def create_prediction_vector(athlete_name, event_categories, profiles_df):

    if athlete_name not in profiles_df.index:
        return None

    feature_columns = profiles_df.columns.tolist()
    athlete_stats = profiles_df.loc[athlete_name]
    
    competition_vector = []
    for col in feature_columns:

        is_relevant = any(cat.lower() in col for cat in event_categories)
        if is_relevant:
            competition_vector.append(athlete_stats[col])
        else:
            competition_vector.append(0)
            
    return competition_vector

@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, scaler, athlete_profiles is not None]):
        return jsonify({'error': 'Model not loaded, check server logs.'}), 500

    data = request.get_json()
    if not data or 'athletes' not in data or 'event_categories' not in data:
        return jsonify({'error': 'Invalid input. Required keys: "athletes", "event_categories"'}), 400

    athlete_names = data['athletes']
    event_categories = data['event_categories']
    
    prediction_vectors = []
    valid_athletes = []
    
    for name in athlete_names:
        vec = create_prediction_vector(name, event_categories, athlete_profiles)
        if vec:
            prediction_vectors.append(vec)
            valid_athletes.append(name)
        else:
            print(f"Warning: Athlete '{name}' not found in profiles.")

    if not valid_athletes:
        return jsonify({'error': 'None of the provided athletes were found.'}), 404



    print("\n--- DEBUGGING RAW PREDICTION VECTORS ---")
    for name, vec in zip(valid_athletes, prediction_vectors):
        print(f"Athlete: {name}")

        print(f"  Raw Vector (first 10 features): {np.round(vec[:10], 4).tolist()}")
    print("----------------------------------------\n")


    X_pred = np.array(prediction_vectors)
    X_pred_scaled = scaler.transform(X_pred)
    predictions = model.predict(X_pred_scaled)

    results = []
    for athlete, pred_score in zip(valid_athletes, predictions.flatten()):
        results.append({'athlete_name': athlete, 'predicted_score': float(pred_score)})

    sorted_results = sorted(results, key=lambda x: x['predicted_score'])

    return jsonify(sorted_results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)