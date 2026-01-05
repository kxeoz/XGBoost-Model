# check_accuracy.py
# Script to inspect model accuracy (R²), best feature, and run cross-validation.
# Run: python check_accuracy.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics import make_scorer
import warnings
warnings.filterwarnings('ignore')  # Suppress minor warnings for clean output

# Copy key parts from app.py for standalone use
CSV_COLUMNS = [
    'STATION', 'DATE_OF_RESPONSE', 'LOCATION', 'RESPONDING_UNIT', 'TIME_RECEIVED',
    'TIME_DISPATCHED', 'TIME_ARRIVAL', 'RESPONSE_TIME_MIN', 'DISTANCE', 'ALARM_STATUS',
    'TIME_LAST_ALARM', 'TYPE_OF_OCCUPANCY', 'INJURED_CIV', 'INJURED_BFP',
    'DEATH_CIV', 'DEATH_BFP', 'REMARKS', 'Temperature_C', 'Humidity_%',
    'Wind_Speed_kmh', 'Precipitation_mm', 'Weather_Condition', 'Road_Condition'
]

def safe_int(value, default=0):
    if value is None or value == '':
        return default
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default

def safe_float(value, default=0.0):
    if value is None or value == '':
        return default
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return default

def determine_severity(row):
    death_civ = safe_int(row.get('DEATH_CIV', 0))
    death_bfp = safe_int(row.get('DEATH_BFP', 0))
    injured_civ = safe_int(row.get('INJURED_CIV', 0))
    injured_bfp = safe_int(row.get('INJURED_BFP', 0))
    
    total_deaths = death_civ + death_bfp
    total_injured = injured_civ + injured_bfp
    
    if total_deaths > 0:
        return 'severe'
    elif total_injured > 0:
        return 'major'
    elif total_injured == 0 and total_deaths == 0:
        return 'moderate'
    else:
        return 'minor'

def load_incidents_from_csv():
    """Load and preprocess incidents from CSV (adapted from app.py)"""
    csv_file = 'fire-incidents.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run app.py to generate/load data.")
        return []
    
    incidents = []
    try:
        df_raw = pd.read_csv(csv_file, usecols=CSV_COLUMNS)  # Use only defined columns
        for _, row in df_raw.iterrows():
            incident_data = {
                'severity': determine_severity(row),
                'type': str(row.get('TYPE_OF_OCCUPANCY', 'Other')),
                'location': row.get('LOCATION', 'Unknown'),
                'weather': row.get('Weather_Condition', 'Unknown'),
                'temperature': safe_float(row.get('Temperature_C', 25)),
                'humidity': safe_float(row.get('Humidity_%', 70)),
                'wind_speed': safe_float(row.get('Wind_Speed_kmh', 10)),
                'response_time': safe_int(row.get('RESPONSE_TIME_MIN', 0)),
                'distance': safe_float(row.get('DISTANCE', 3.0)),
                'date': row.get('DATE_OF_RESPONSE', ''),
                'station': row.get('STATION', ''),
                'road_condition': row.get('Road_Condition', '')
            }
            incidents.append(incident_data)
        print(f"Loaded {len(incidents)} incidents from CSV.")
        return incidents
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

class SimpleAnalyzer:
    """Minimal analyzer for inspection (copies essentials from app.py)"""
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features = ['severity', 'type', 'temperature', 'humidity', 'wind_speed', 'distance']
        self.target = 'response_time'
        self._baseline_performance = None

    def load_model(self):
        """Load saved model if exists"""
        pkl_file = 'fire_incident_analyzer.pkl'
        if os.path.exists(pkl_file):
            try:
                data = joblib.load(pkl_file)
                self.model = data['model']
                self.label_encoders = data.get('label_encoders', {})
                self._baseline_performance = data.get('baseline_performance')
                print(f"Loaded model from {pkl_file}.")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print(f"No model file found at {pkl_file}. Run app.py to train.")
        return False

    def preprocess_data(self, df):
        """Preprocess for features"""
        df_processed = df.copy()
        categorical_columns = ['severity', 'type']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                try:
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
                except ValueError:
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
        
        # Ensure numerics
        numeric_features = ['temperature', 'humidity', 'wind_speed', 'response_time', 'distance']
        for col in numeric_features:
            if col in df_processed.columns:
                median_val = df_processed[col].median()
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce').fillna(median_val)
        
        return df_processed

    def get_current_accuracy_and_feature(self):
        """Check loaded model's accuracy and best feature"""
        if not self.model or not self._baseline_performance:
            print("No loaded model or performance data.")
            return
        
        r2 = self._baseline_performance.get('r2', 0)
        mae = self._baseline_performance.get('mae', 0)
        print(f"\n=== Loaded Model Metrics ===")
        print(f"Current Accuracy (R²): {r2:.3f}")
        print(f"MAE (Mean Absolute Error): {mae:.3f} minutes")
        print(f"Training Samples: {self._baseline_performance.get('training_samples', 'N/A')}")
        
        # Best Feature
        if hasattr(self.model, 'feature_importances_'):
            importances = dict(zip(self.features, self.model.feature_importances_))
            best_feature = max(importances, key=importances.get)
            best_score = importances[best_feature]
            print(f"\nBest Feature: {best_feature} (Importance: {best_score:.3f})")
            print("All Feature Importances:")
            for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {imp:.3f}")
        else:
            print("No feature importances available.")

    def run_cross_validation(self, incidents):
        """Run 5-fold CV for robust accuracy check"""
        if len(incidents) < 10:
            print("Not enough data for CV (need 10+ incidents).")
            return
        
        df = pd.DataFrame(incidents)
        df_processed = self.preprocess_data(df)
        X = df_processed[self.features]
        y = df_processed[self.target]
        
        model_cv = RandomForestRegressor(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(model_cv, X, y, cv=5, scoring='r2')
        cv_r2_mean = cv_scores.mean()
        cv_r2_std = cv_scores.std()
        
        mae_scorer = make_scorer(mean_absolute_error)
        cv_mae = cross_val_score(model_cv, X, y, cv=5, scoring=mae_scorer)
        cv_mae_mean = cv_mae.mean()
        
        print(f"\n=== Cross-Validation Results (5-Fold) ===")
        print(f"CV R² Mean: {cv_r2_mean:.3f} ± {cv_r2_std:.3f}")
        print(f"CV MAE Mean: {cv_mae_mean:.3f} minutes")
        print("Individual CV R² Scores:", [f"{s:.3f}" for s in cv_scores])

    def run_holdout_test(self, incidents):
        """Run 80/20 train-test split for out-of-sample check"""
        if len(incidents) < 10:
            print("Not enough data for holdout test.")
            return
        
        df = pd.DataFrame(incidents)
        df_processed = self.preprocess_data(df)
        X = df_processed[self.features]
        y = df_processed[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_test = RandomForestRegressor(n_estimators=100, random_state=42)
        model_test.fit(X_train, y_train)
        
        y_pred = model_test.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt((y_test - y_pred)**2).mean()  # RMSE
        
        print(f"\n=== Hold-Out Test Set Results (20% Test) ===")
        print(f"Test R²: {r2:.3f}")
        print(f"Test MAE: {mae:.3f} minutes")
        print(f"Test RMSE: {rmse:.3f} minutes")

if __name__ == '__main__':
    print("Fire Incident Model Inspector - October 24, 2025")
    print("=" * 50)
    
    analyzer = SimpleAnalyzer()
    
    # Load model and check basics
    if analyzer.load_model():
        analyzer.get_current_accuracy_and_feature()
    
    # Load data and run advanced checks
    incidents = load_incidents_from_csv()
    if incidents:
        analyzer.run_cross_validation(incidents)
        analyzer.run_holdout_test(incidents)
    else:
        print("No incidents loaded; skipping CV and holdout checks.")
    
    print("\nDone! Review outputs above for model health.")