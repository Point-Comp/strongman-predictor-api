# 2_model_training.py (Final)
import pandas as pd
import numpy as np
import tensorflow as tf
import random 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


def create_training_set(cleaned_data, athlete_profiles):
    
    feature_columns = athlete_profiles.columns.tolist()
    X_data = []
    y_data = []

    competitions = cleaned_data.groupby(['contest', 'date'])
    
    for (contest_name, contest_date), contest_df in competitions:
        event_categories_in_contest = contest_df['event_category'].unique()
        
        for athlete_name in contest_df['athlete_name'].unique():
            if athlete_name not in athlete_profiles.index:
                continue

            competition_vector = []
            athlete_stats = athlete_profiles.loc[athlete_name]
            
            for col in feature_columns:
                is_relevant = any(cat_keyword in col for cat_keyword in event_categories_in_contest)
                if is_relevant:
                    competition_vector.append(athlete_stats[col])
                else:
                    competition_vector.append(0)

            X_data.append(competition_vector)
            final_placing = contest_df[contest_df['athlete_name'] == athlete_name]['final_placing'].iloc[0]
            y_data.append(final_placing)

    return np.array(X_data), np.array(y_data)

def run_training(cleaned_data_path, profile_path, model_save_path, scaler_save_path):

    print("Step 1: Loading preprocessed data...")
    cleaned_data = pd.read_csv(cleaned_data_path)
    athlete_profiles = pd.read_csv(profile_path, index_col='athlete_name')

    print("Step 2: Creating training set...")
    X, y = create_training_set(cleaned_data, athlete_profiles)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Step 3: Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Step 4: Building and training Keras model...")
    model = keras.Sequential([
        layers.Input(shape=(X_train_scaled.shape[1],)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1)
    ])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_absolute_error')
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    model.fit(
        X_train_scaled, y_train,
        epochs=150,
        batch_size=64,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Step 5: Evaluating and saving the new model...")
    loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Set Mean Absolute Error: {loss:.2f} places")

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)

    print(f"Model saved to '{model_save_path}'")
    print(f"Data scaler saved to '{scaler_save_path}'")

if __name__ == '__main__':
    CLEANED_DATA_CSV = 'cleaned_strongman_data.csv'
    ATHLETE_PROFILES_CSV = 'athlete_profiles.csv'
    MODEL_PATH = 'strongman_predictor.h5'
    SCALER_PATH = 'data_scaler.pkl'
    
    run_training(CLEANED_DATA_CSV, ATHLETE_PROFILES_CSV, MODEL_PATH, SCALER_PATH)
