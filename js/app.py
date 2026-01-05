# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
import joblib
import json
from datetime import datetime, timedelta
import os
import csv

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

app = Flask(
    __name__,
    template_folder=ROOT_DIR,
    static_folder=ROOT_DIR,
    static_url_path=''  # serve existing css/js paths without rewriting
)
CORS(app)
# Define CSV columns
CSV_COLUMNS = [
    'STATION', 'DATE_OF_RESPONSE', 'LOCATION', 'RESPONDING_UNIT', 'TIME_RECEIVED',
    'TIME_DISPATCHED', 'TIME_ARRIVAL', 'RESPONSE_TIME_MIN', 'DISTANCE', 'ALARM_STATUS',
    'TIME_LAST_ALARM', 'TYPE_OF_OCCUPANCY', 'INJURED_CIV', 'INJURED_BFP',
    'DEATH_CIV', 'DEATH_BFP', 'REMARKS', 'Temperature_C', 'Humidity_%',
    'Wind_Speed_kmh', 'Precipitation_mm', 'Weather_Condition', 'Road_Condition'
]
# Global variables
fire_analyzer = None
fire_incidents = []
fire_hydrants = []
hazard_roads = []
# JSON Serialization Helper
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)
app.json_encoder = NumpyEncoder
# Root route renders the main dashboard without needing a separate live server
@app.route('/')
def serve_dashboard():
    return render_template('dashboard.html')
# Helper functions
def safe_int(value, default=0):
    """Safely convert to int"""
    if value is None or value == '':
        return default
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return default
def safe_float(value, default=0.0):
    """Safely convert to float"""
    if value is None or value == '':
        return default
    try:
        return float(str(value))
    except (ValueError, TypeError):
        return default
def parse_date(date_str):
    """Parse various date formats"""
    try:
        if not date_str:
            return datetime.now().isoformat()
        for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y']:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        return datetime.now().isoformat()
    except:
        return datetime.now().isoformat()
def determine_severity(row):
    """Determine severity based on injuries, deaths"""
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
    """Load incidents from CSV file"""
    global fire_incidents
    fire_incidents = []
    csv_file = os.path.join(SCRIPT_DIR, 'fire-incidents.csv')
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found.")
        return
    loaded_count = 0
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
      
            for row in csv_reader:
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
                    'alarm_status': row.get('ALARM_STATUS', ''),
                    'injured_civ': safe_int(row.get('INJURED_CIV', 0)),
                    'injured_bfp': safe_int(row.get('INJURED_BFP', 0)),
                    'death_civ': safe_int(row.get('DEATH_CIV', 0)),
                    'death_bfp': safe_int(row.get('DEATH_BFP', 0)),
                    'station': row.get('STATION', ''),
                    'road_condition': row.get('Road_Condition', '')
                }
          
                incident_id = len(fire_incidents) + 1
                incident_data['id'] = incident_id
                incident_data['timestamp'] = parse_date(row.get('DATE_OF_RESPONSE', ''))
          
                fire_incidents.append(incident_data)
                loaded_count += 1
        print(f"Loaded {loaded_count} incidents from CSV.")
    except Exception as e:
        print(f"Error loading CSV: {e}")
def append_to_csv(incident_data):
    """Append new incident to CSV with proper data formatting"""
    csv_file = 'fire-incidents.csv'
    # Calculate response time from times if available
    response_time = incident_data.get('response_time_min', 0)
    if not response_time and all(k in incident_data for k in ['time_received', 'time_arrival']):
        try:
            time_format = '%H:%M'
            received = datetime.strptime(incident_data['time_received'], time_format)
            arrival = datetime.strptime(incident_data['time_arrival'], time_format)
            response_time = (arrival - received).total_seconds() / 60
        except:
            response_time = 5 # default fallback
    # Format date properly
    date_of_response = incident_data.get('date_of_response', '')
    if not date_of_response:
        date_of_response = datetime.now().strftime('%m/%d/%Y')
    else:
        try:
            # Convert from YYYY-MM-DD to MM/DD/YYYY
            dt = datetime.strptime(date_of_response, '%Y-%m-%d')
            date_of_response = dt.strftime('%m/%d/%Y')
        except:
            date_of_response = datetime.now().strftime('%m/%d/%Y')
    row = {
        'STATION': incident_data.get('station', 'Santa Cruz, Laguna'),
        'DATE_OF_RESPONSE': date_of_response,
        'LOCATION': incident_data.get('location', 'Unknown'),
        'RESPONDING_UNIT': incident_data.get('responding_unit', 'Shift B'),
        'TIME_RECEIVED': incident_data.get('time_received', '08:00'),
        'TIME_DISPATCHED': incident_data.get('time_dispatched', '08:00'),
        'TIME_ARRIVAL': incident_data.get('time_arrival', '08:05'),
        'RESPONSE_TIME_MIN': response_time,
        'DISTANCE': safe_float(incident_data.get('distance', 3.3)),
        'ALARM_STATUS': incident_data.get('alarm_status', 'FA'),
        'TIME_LAST_ALARM': f"{date_of_response} {incident_data.get('time_received', '08:00')}",
        'TYPE_OF_OCCUPANCY': incident_data.get('type_of_occupancy', 'Other'),
        'INJURED_CIV': safe_int(incident_data.get('injured_civ', 0)),
        'INJURED_BFP': safe_int(incident_data.get('injured_bfp', 0)),
        'DEATH_CIV': safe_int(incident_data.get('death_civ', 0)),
        'DEATH_BFP': safe_int(incident_data.get('death_bfp', 0)),
        'REMARKS': incident_data.get('remarks', 'Case Closed'),
        'Temperature_C': safe_float(incident_data.get('temperature_c', 25.0)),
        'Humidity_%': safe_int(incident_data.get('humidity_pct', 70)),
        'Wind_Speed_kmh': safe_float(incident_data.get('wind_speed_kmh', 10.0)),
        'Precipitation_mm': safe_float(incident_data.get('precipitation_mm', 0.0)),
        'Weather_Condition': incident_data.get('weather_condition', 'Sunny'),
        'Road_Condition': incident_data.get('road_condition', 'Dry')
    }
    try:
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended new incident to CSV: {row['DATE_OF_RESPONSE']} - {row['LOCATION']}")
        return True
    except Exception as e:
        print(f"Error appending to CSV: {str(e)}")
        return False
def get_initial_data_suggestions():
    """Suggestions when there's not enough data"""
    return [{
        'category': 'data_collection',
        'priority': 'high',
        'title': '📊 Build Your Response Database',
        'description': 'Start collecting incident data to enable performance analysis.',
        'actionable_steps': [
            'Record every fire response with accurate timing',
            'Include distance measurements and location details',
            'Document weather and road conditions',
            'Track response challenges and outcomes'
        ],
        'expected_impact': 'Enable data-driven performance improvement',
        'implementation_difficulty': 'low',
        'time_to_implement': 'Immediate'
    }]
def generate_comprehensive_feedback(incident_data, comparison, historical_data):
    """Generate comprehensive feedback with performance analysis and improvement suggestions"""
   
    # Extract response time from incident data
    current_time = incident_data.get('response_time', incident_data.get('response_time_min', 0))
    expected_time = comparison.get('similar_average', current_time)
    time_diff = comparison.get('time_difference', 0)
   
    # Performance analysis
    performance_analysis = analyze_performance(current_time, expected_time, time_diff, incident_data.get('distance', 0))
   
    # Improvement suggestions
    improvement_suggestions = generate_improvement_suggestions_detailed(incident_data, comparison, historical_data)
   
    # Training recommendations
    training_recommendations = generate_training_recommendations_detailed(historical_data)
   
    # Success factors (what went well)
    success_factors = identify_success_factors(incident_data, comparison)
   
    # Get model accuracy
    model_accuracy = 0
    if fire_analyzer and hasattr(fire_analyzer, '_baseline_performance'):
        baseline = fire_analyzer._baseline_performance
        if baseline and 'regression' in baseline:
            model_accuracy = baseline['regression'].get('r2_score', 0)
    
    return {
        'performance_overview': {
            'avg_response_time': sum(inc.get('response_time', 0) for inc in historical_data) / len(historical_data) if historical_data else 0,
            'model_accuracy': model_accuracy,
            'total_incidents': len(historical_data)
        },
        'predicted_vs_actual': {
            'predicted': comparison.get('prediction', {}).get('predicted_response_time', 0),
            'actual': current_time,
            'difference': current_time - comparison.get('prediction', {}).get('predicted_response_time', 0)
        },
        'performance_analysis': performance_analysis,
        'improvement_suggestions': [s.get('description') or s.get('title', str(s)) for s in improvement_suggestions],
        'training_recommendations': [r.get('description') or r.get('title', str(r)) for r in training_recommendations],
        'success_factors': success_factors,
        'contextual_factors': {
            'fire_type': incident_data.get('type', 'N/A'),
            'time_of_day': incident_data.get('time_received', 'N/A'),
            'weather_condition': incident_data.get('weather', 'N/A'),
            'distance': incident_data.get('distance', 0)
        },
        'comparison_metrics': {
            'current_response_time': current_time,
            'expected_response_time': round(expected_time, 1),
            'time_difference': round(time_diff, 1),
            'performance_ratio': comparison.get('performance_ratio', 1.0),
            'similar_incidents_count': comparison.get('similar_count', 0)
        }
    }
def analyze_performance(current_time, expected_time, time_diff, distance):
    """Analyze performance and provide detailed feedback"""
    performance_ratio = current_time / expected_time if expected_time > 0 else 1.0
    if performance_ratio < 0.8:
        status = "excellent"
        message = "🚀 Outstanding performance! Significantly faster than expected."
        color = "green"
    elif performance_ratio < 1.0:
        status = "good"
        message = "✅ Good performance! Faster than average response."
        color = "blue"
    elif performance_ratio < 1.2:
        status = "average"
        message = "⚠️ Average performance. Meets expectations."
        color = "yellow"
    elif performance_ratio < 1.5:
        status = "needs_improvement"
        message = "📊 Below average performance. Room for improvement."
        color = "orange"
    else:
        status = "poor"
        message = "🚨 Significant delay detected. Immediate improvement needed."
        color = "red"
    # Efficiency analysis
    efficiency = current_time / distance if distance > 0 else 0
    efficiency_rating = "high" if efficiency < 2.0 else "medium" if efficiency < 3.0 else "low"
    return {
        'status': status,
        'message': message,
        'color': color,
        'efficiency_rating': efficiency_rating,
        'efficiency_score': round(efficiency, 2),
        'performance_ratio': round(performance_ratio, 2),
        'improvement_opportunity': max(0, time_diff)
    }
def generate_improvement_suggestions_detailed(incident_data, comparison, historical_data):
    """Generate detailed, actionable improvement suggestions"""
    suggestions = []
    current_time = incident_data.get('response_time', 0)
    expected_time = comparison.get('similar_average', current_time)
    time_diff = comparison.get('time_difference', 0)
    distance = incident_data.get('distance', 0)
    # 1. Response Time Suggestions
    if time_diff > 5:
        suggestions.append({
            'category': 'response_time',
            'priority': 'high',
            'title': '🚨 Reduce Response Time',
            'description': f'Your response was {time_diff:.1f} minutes slower than similar incidents.',
            'actionable_steps': [
                'Review dispatch-to-departure procedures',
                'Optimize vehicle readiness checks',
                'Implement pre-planned routes for common locations',
                'Conduct time-motion studies on station exit'
            ],
            'expected_impact': f'Reduce response time by {min(time_diff, 8):.1f} minutes',
            'implementation_difficulty': 'medium',
            'time_to_implement': '2-4 weeks'
        })
    # 2. Distance Efficiency Suggestions
    if distance > 0:
        efficiency = current_time / distance
        if efficiency > 3.0: # More than 3 minutes per km
            suggestions.append({
                'category': 'route_efficiency',
                'priority': 'high',
                'title': '🛣️ Improve Route Efficiency',
                'description': f'Travel efficiency of {efficiency:.1f} min/km is below optimal.',
                'actionable_steps': [
                    'Analyze GPS data for route deviations',
                    'Train drivers on optimal speed maintenance',
                    'Coordinate with traffic management for priority routing',
                    'Review road condition reports before dispatch'
                ],
                'expected_impact': 'Improve efficiency by 15-25%',
                'implementation_difficulty': 'low',
                'time_to_implement': '1-2 weeks'
            })
    # 3. Weather Adaptation Suggestions
    weather = incident_data.get('weather', '').lower()
    adverse_weather = any(term in weather for term in ['rain', 'storm', 'typhoon', 'heavy'])
    if adverse_weather and time_diff > 2:
        suggestions.append({
            'category': 'weather_adaptation',
            'priority': 'medium',
            'title': '🌧️ Enhance Weather Response',
            'description': 'Adverse weather conditions impacted response time.',
            'actionable_steps': [
                'Develop weather-specific response protocols',
                'Pre-position vehicles during storm warnings',
                'Train drivers on wet weather driving techniques',
                'Install weather monitoring alerts in dispatch'
            ],
            'expected_impact': 'Reduce weather-related delays by 30%',
            'implementation_difficulty': 'medium',
            'time_to_implement': '3-5 weeks'
        })
    # 4. Equipment Optimization
    if current_time > 15: # Long responses
        suggestions.append({
            'category': 'equipment',
            'priority': 'medium',
            'title': '🚛 Optimize Equipment Deployment',
            'description': 'Long response times indicate potential equipment optimization opportunities.',
            'actionable_steps': [
                'Consider satellite station deployment',
                'Review vehicle maintenance schedules',
                'Implement mobile equipment pre-loading',
                'Develop mutual aid response coordination'
            ],
            'expected_impact': 'Improve long-distance response efficiency',
            'implementation_difficulty': 'high',
            'time_to_implement': '1-3 months'
        })
    # 5. Training and Preparedness
    if len(historical_data) < 20:
        suggestions.append({
            'category': 'training',
            'priority': 'low',
            'title': '📚 Enhance Training Program',
            'description': 'Limited historical data suggests expanding training scenarios.',
            'actionable_steps': [
                'Develop scenario-based training exercises',
                'Record and analyze all response metrics',
                'Implement regular performance reviews',
                'Create incident response playbooks'
            ],
            'expected_impact': 'Build comprehensive response database',
            'implementation_difficulty': 'low',
            'time_to_implement': 'Ongoing'
        })
    return suggestions
def generate_training_recommendations_detailed(historical_data):
    """Generate detailed training recommendations"""
    recommendations = []
    if len(historical_data) < 10:
        recommendations.append({
            'type': 'data_collection',
            'priority': 'high',
            'title': '📈 Expand Data Collection',
            'description': f'Currently have {len(historical_data)} incidents. Aim for 20+ for reliable analysis.',
            'actions': [
                'Ensure all responses are recorded in detail',
                'Include varied incident types and locations',
                'Track environmental and road conditions',
                'Document response challenges and successes'
            ],
            'benefits': 'More accurate performance predictions and better improvement suggestions'
        })
    # Analyze response time distribution
    df = pd.DataFrame(historical_data)
    if 'response_time' in df.columns:
        avg_response = df['response_time'].mean()
        response_std = df['response_time'].std()
    
        if response_std > 5: # High variability
            recommendations.append({
                'type': 'consistency_training',
                'priority': 'medium',
                'title': '🎯 Improve Response Consistency',
                'description': 'High variability in response times indicates inconsistent performance.',
                'actions': [
                    'Standardize response procedures',
                    'Implement crew rotation analysis',
                    'Develop performance benchmarks',
                    'Create response time targets by incident type'
                ],
                'benefits': 'More predictable and reliable response performance'
            })
    return recommendations
def identify_success_factors(incident_data, comparison):
    """Identify what worked well in the response"""
    success_factors = []
    current_time = incident_data.get('response_time', 0)
    expected_time = comparison.get('similar_average', current_time)
    time_diff = comparison.get('time_difference', 0)
    # What went well
    if time_diff < -2: # Faster than expected
        success_factors.append({
            'factor': 'exceptional_speed',
            'message': f'🚀 Response was {abs(time_diff):.1f} minutes faster than similar incidents!',
            'best_practices': [
                'Document the strategies used in this response',
                'Share successful approaches with other teams',
                'Consider making these practices standard',
                'Analyze what made this response particularly efficient'
            ]
        })
    # Efficient distance handling
    distance = incident_data.get('distance', 0)
    if distance > 0 and current_time > 0:
        efficiency = current_time / distance
        if efficiency < 2.0: # Very efficient
            success_factors.append({
                'factor': 'route_efficiency',
                'message': f'✅ Excellent travel efficiency: {efficiency:.1f} minutes per kilometer',
                'best_practices': [
                    'Maintain current route planning approach',
                    'Continue current driver training protocols',
                    'Document the navigation strategies used',
                    'Share efficient route knowledge'
                ]
            })
    # Good weather adaptation
    weather = incident_data.get('weather', '').lower()
    adverse_weather = any(term in weather for term in ['rain', 'storm', 'typhoon', 'heavy'])
    if adverse_weather and time_diff <= 2:
        success_factors.append({
            'factor': 'weather_adaptation',
            'message': '🌧️ Effective response despite adverse weather conditions',
            'best_practices': [
                'Continue current weather response protocols',
                'Maintain vehicle readiness for all conditions',
                'Keep driver training on adverse weather handling',
                'Document weather-specific successful strategies'
            ]
        })
    return success_factors
# Enhanced Analyzer Class
class FireIncidentAnalyzer:
    def __init__(self):
        self.regressor = None
        self.classifier = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.regression_features = ['severity', 'type', 'temperature', 'humidity', 'wind_speed', 'distance', 'weather_impact', 'time_of_day']
        self.classification_features = ['severity', 'type', 'temperature', 'humidity', 'wind_speed', 'distance', 'weather_impact', 'time_of_day', 'is_weekend']
        self.regression_target = 'response_time'
        self.classification_target = 'response_category'
        self._last_analysis = None
        self._baseline_performance = None
        self.load_analyzer()
  
    def load_analyzer(self):
        """Load analyzer from file if exists"""
        try:
            if os.path.exists('fire_incident_analyzer.pkl'):
                analyzer_data = joblib.load('fire_incident_analyzer.pkl')
                self.regressor = analyzer_data['regressor']
                self.classifier = analyzer_data['classifier']
                self.label_encoders = analyzer_data['label_encoders']
                self.scaler = analyzer_data.get('scaler', StandardScaler())
                self._baseline_performance = analyzer_data.get('baseline_performance')
                self._last_analysis = analyzer_data.get('last_analysis')
                print("Analyzer loaded from fire_incident_analyzer.pkl")
            else:
                print("No saved analyzer found; will train fresh")
        except Exception as e:
            print(f"Error loading analyzer: {e}")
    def create_response_categories(self, response_times):
        """Convert response times to categories for classification"""
        categories = []
        for time in response_times:
            if time <= 5:
                categories.append('very_fast')
            elif time <= 10:
                categories.append('fast')
            elif time <= 15:
                categories.append('moderate')
            elif time <= 20:
                categories.append('slow')
            else:
                categories.append('very_slow')
        return categories
    def engineer_features(self, df):
        """Add engineered features for better prediction with consistent output"""
        df_engineered = df.copy()
    
        # Convert timestamp to features - FIXED DATETIME PARSING
        if 'timestamp' in df_engineered.columns:
            try:
                # Handle various timestamp formats including fractional seconds
                df_engineered['timestamp'] = pd.to_datetime(
                    df_engineered['timestamp'],
                    errors='coerce',
                    utc=True
                ).dt.tz_convert(None) # Convert to naive datetime
            
                # Extract time-based features
                df_engineered['time_of_day'] = df_engineered['timestamp'].dt.hour.fillna(12)
                df_engineered['day_of_week'] = df_engineered['timestamp'].dt.dayofweek.fillna(0)
                df_engineered['is_weekend'] = df_engineered['day_of_week'].isin([5, 6]).astype(int)
                df_engineered['month'] = df_engineered['timestamp'].dt.month.fillna(1)
                df_engineered['is_peak_hours'] = df_engineered['time_of_day'].between(7, 9).astype(int)
            except Exception as e:
                print(f"Warning: Error processing timestamps: {e}")
                # Set default values if timestamp processing fails
                df_engineered['time_of_day'] = 12
                df_engineered['day_of_week'] = 0
                df_engineered['is_weekend'] = 0
                df_engineered['month'] = 1
                df_engineered['is_peak_hours'] = 0
        else:
            # Add default time features if timestamp is missing
            df_engineered['time_of_day'] = 12
            df_engineered['day_of_week'] = 0
            df_engineered['is_weekend'] = 0
            df_engineered['month'] = 1
            df_engineered['is_peak_hours'] = 0
    
        # Weather impact score
        weather_impact = {
            'Sunny': 1.0, 'Clear': 1.0, 'Partly Cloudy': 1.1,
            'Cloudy': 1.2, 'Rainy': 1.5, 'Heavy Rain': 1.8,
            'Stormy': 2.0, 'Typhoon': 2.5
        }
        if 'weather' in df_engineered.columns:
            df_engineered['weather_impact'] = df_engineered['weather'].map(weather_impact).fillna(1.2)
        else:
            df_engineered['weather_impact'] = 1.2
    
        # Severity weight
        severity_weight = {'minor': 1.0, 'moderate': 1.3, 'major': 1.7, 'severe': 2.2}
        if 'severity' in df_engineered.columns:
            df_engineered['severity_weight'] = df_engineered['severity'].map(severity_weight).fillna(1.0)
        else:
            df_engineered['severity_weight'] = 1.0
    
        # Road condition impact
        road_impact = {'Dry': 1.0, 'Wet': 1.3, 'Flooded': 1.8, 'Icy': 2.0}
        if 'road_condition' in df_engineered.columns:
            df_engineered['road_impact'] = df_engineered['road_condition'].map(road_impact).fillna(1.0)
        else:
            df_engineered['road_impact'] = 1.0
    
        # Ensure all expected features exist
        expected_features = [
            'severity', 'type', 'temperature', 'humidity', 'wind_speed', 'distance',
            'weather_impact', 'time_of_day', 'severity_weight', 'road_impact',
            'day_of_week', 'is_weekend', 'month', 'is_peak_hours'
        ]
    
        for feature in expected_features:
            if feature not in df_engineered.columns:
                if feature in ['time_of_day', 'day_of_week', 'is_weekend', 'month', 'is_peak_hours']:
                    df_engineered[feature] = 0
                else:
                    df_engineered[feature] = 1.0
    
        return df_engineered
    def preprocess_data(self, df, for_classification=False):
        """Preprocess the data for analysis"""
        df_processed = df.copy()
  
        # Feature engineering
        df_processed = self.engineer_features(df_processed)
  
        # Handle categorical variables
        categorical_columns = ['severity', 'type', 'weather']
        for col in categorical_columns:
            if col in df_processed.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                try:
                    # Handle new categories during prediction
                    mask = df_processed[col].isin(self.label_encoders[col].classes_)
                    if not mask.all():
                        # For new categories, use the most frequent class
                        most_frequent = df_processed[col].mode()[0] if not df_processed[col].empty else 'Unknown'
                        df_processed.loc[~mask, col] = most_frequent
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col])
                except (ValueError, AttributeError):
                    # Refit encoder if needed
                    self.label_encoders[col] = LabelEncoder()
                    df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
  
        # Ensure all required features exist
        required_features = self.regression_features if not for_classification else self.classification_features
        for feature in required_features:
            if feature not in df_processed.columns:
                # Add missing features with default values
                if feature in ['weather_impact', 'severity_weight', 'road_impact']:
                    df_processed[feature] = 1.0
                elif feature in ['time_of_day', 'day_of_week', 'is_weekend', 'month', 'is_peak_hours']:
                    df_processed[feature] = 0
                else:
                    df_processed[feature] = 0.0
  
        # Scale numerical features for classification
        if for_classification:
            numerical_features = ['temperature', 'humidity', 'wind_speed', 'distance', 'time_of_day']
            for col in numerical_features:
                if col in df_processed.columns:
                    df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                    df_processed[col] = df_processed[col].fillna(df_processed[col].median())
      
            # Fit scaler only on training data
            if hasattr(self, '_is_training') and self._is_training:
                df_processed[numerical_features] = self.scaler.fit_transform(df_processed[numerical_features])
            else:
                df_processed[numerical_features] = self.scaler.transform(df_processed[numerical_features])
  
        return df_processed
    def select_important_features(self, X, y, feature_names, top_k=5):
        """Select only the most important features to reduce overfitting"""
        is_classification = len(np.unique(y)) < len(y) * 0.1 # Rough heuristic
  
        if is_classification:
            selector = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(top_k, X.shape[1]))
  
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]
  
        return X_selected, selected_features
    def train_analyzer(self, incidents_data):
        """Train both regressor and classifier with reduced overfitting"""
        try:
            print(f"Starting enhanced analysis training with {len(incidents_data)} incidents")
            df = pd.DataFrame(incidents_data)
      
            # Clean and prepare data
            df = self.clean_data(df)
      
            # Set training flag for scaler
            self._is_training = True
      
            # Train Regression Model with reduced complexity
            print("Training regression model...")
            df_regression = self.preprocess_data(df, for_classification=False)
            X_reg = df_regression[self.regression_features]
            y_reg = df_regression[self.regression_target]
      
            # Select important features
            X_reg_selected, selected_reg_features = self.select_important_features(
                X_reg, y_reg, self.regression_features, top_k=5
            )
            self.regression_features = selected_reg_features # Update to use only important features
      
            # SIMPLIFIED REGRESSOR - Reduced complexity
            self.regressor = RandomForestRegressor(
                n_estimators=50, # Reduced from 200
                max_depth=8, # Reduced from 15
                min_samples_split=10, # Increased from 5
                min_samples_leaf=4, # Increased from 2
                max_features='sqrt', # Limit features per tree
                random_state=42
            )
            self.regressor.fit(X_reg_selected, y_reg)
      
            # Calculate regression performance with cross-validation
            cv_reg_scores = cross_val_score(self.regressor, X_reg_selected, y_reg, cv=5, scoring='r2')
            y_reg_pred = self.regressor.predict(X_reg_selected)
            r2 = r2_score(y_reg, y_reg_pred)
            mae = mean_absolute_error(y_reg, y_reg_pred)
      
            # Train Classification Model with reduced complexity
            print("Training classification model...")
            df['response_category'] = self.create_response_categories(df['response_time'])
            df_classification = self.preprocess_data(df, for_classification=True)
            X_clf = df_classification[self.classification_features]
            y_clf = df_classification[self.classification_target]
      
            # Select important features
            X_clf_selected, selected_clf_features = self.select_important_features(
                X_clf, y_clf, self.classification_features, top_k=15
            )
            self.classification_features = selected_clf_features

            # ENHANCED CLASSIFIER - Increased complexity for better accuracy
            self.classifier = RandomForestClassifier(
                n_estimators=300, # Increased from 50
                max_depth=20, # Increased from 6
                min_samples_split=2, # Reduced from 8
                min_samples_leaf=1, # Reduced from 3
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            )
            self.classifier.fit(X_clf_selected, y_clf)
      
            # Calculate classification performance with cross-validation
            cv_clf_scores = cross_val_score(self.classifier, X_clf_selected, y_clf, cv=5, scoring='accuracy')
            y_clf_pred = self.classifier.predict(X_clf_selected)
            accuracy = accuracy_score(y_clf, y_clf_pred)
      
            # Feature importance - CONVERT TO NATIVE PYTHON TYPES
            reg_feature_importance = dict(zip(self.regression_features, [float(x) for x in self.regressor.feature_importances_]))
            clf_feature_importance = dict(zip(self.classification_features, [float(x) for x in self.classifier.feature_importances_]))
      
            self._baseline_performance = {
                'regression': {
                    'r2': float(r2),
                    'mae': float(mae),
                    'cv_mean_r2': float(cv_reg_scores.mean()),
                    'cv_std_r2': float(cv_reg_scores.std())
                },
                'classification': {
                    'accuracy': float(accuracy),
                    'cv_mean_accuracy': float(cv_clf_scores.mean()),
                    'cv_std_accuracy': float(cv_clf_scores.std()),
                    'class_distribution': {str(k): int(v) for k, v in dict(pd.Series(y_clf).value_counts()).items()}
                },
                'feature_importance': {
                    'regression': reg_feature_importance,
                    'classification': clf_feature_importance
                },
                'training_samples': len(X_reg),
                'timestamp': datetime.now().isoformat()
            }
      
            # Clear training flag
            self._is_training = False
      
            self.save_analyzer()
            print("Enhanced analysis training completed successfully")
            print(f"Classification Accuracy: {accuracy:.4f}")
            print(f"Cross-validation Accuracy: {cv_clf_scores.mean():.4f} (+/- {cv_clf_scores.std() * 2:.4f})")
      
            return self._baseline_performance
      
        except Exception as e:
            print(f"Training exception: {e}")
            return {'error': str(e)}
    def clean_data(self, df):
        """Clean and prepare data for training"""
        df_clean = df.copy()
  
        # Ensure numeric columns
        numeric_features = ['temperature', 'humidity', 'wind_speed', 'response_time', 'distance']
        for col in numeric_features:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].fillna(df_clean[col].median() if not df_clean[col].empty else 0)
  
        # Remove rows with missing response time
        df_clean = df_clean.dropna(subset=['response_time'])
  
        # Handle timestamp issues
        if 'timestamp' in df_clean.columns:
            # Convert to datetime, handling various formats and errors
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce', utc=True)
            # Remove rows with invalid timestamps
            df_clean = df_clean.dropna(subset=['timestamp'])
            # Convert to naive datetime
            df_clean['timestamp'] = df_clean['timestamp'].dt.tz_convert(None)
  
        # Ensure minimum data
        if len(df_clean) < 5:
            raise ValueError("Insufficient data for training (need at least 5 incidents)")
  
        return df_clean
    def validate_model_stability(self, X, y, model_type='regression'):
        """Validate model stability to detect overfitting"""
        from sklearn.model_selection import cross_validate
  
        if model_type == 'regression':
            model = self.regressor
            scoring = ['r2', 'neg_mean_absolute_error']
        else:
            model = self.classifier
            scoring = ['accuracy', 'f1_macro']
  
        # Perform cross-validation
        cv_results = cross_validate(model, X, y, cv=min(5, len(X)), scoring=scoring, return_train_score=True)
  
        # Check for overfitting (train score much higher than test score)
        if 'train_r2' in cv_results and 'test_r2' in cv_results:
            train_test_gap = np.mean(cv_results['train_r2']) - np.mean(cv_results['test_r2'])
            if train_test_gap > 0.3: # Large gap indicates overfitting
                print(f"Warning: Possible overfitting detected. Train-test gap: {train_test_gap:.3f}")
                return False
  
        return True
    def check_model_health(self, X, y, model_type='regression'):
        """Check model health and detect potential overfitting"""
        if model_type == 'regression':
            model = self.regressor
            y_pred = model.predict(X)
            train_score = r2_score(y, y_pred)
       
            # Simple overfitting check
            if train_score > 0.95 and len(X) < 100:
                return {
                    'status': 'warning',
                    'message': 'High R² on small dataset - potential overfitting',
                    'score': train_score
                }
        else:
            model = self.classifier
            y_pred = model.predict(X)
            train_score = accuracy_score(y, y_pred)
       
            if train_score > 0.95 and len(X) < 100:
                return {
                    'status': 'warning',
                    'message': 'High accuracy on small dataset - potential overfitting',
                    'score': train_score
                }
   
        return {
            'status': 'healthy',
            'score': train_score
        }
    def save_analyzer(self):
        """Save analyzer to file"""
        try:
            analyzer_data = {
                'regressor': self.regressor,
                'classifier': self.classifier,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'baseline_performance': self._baseline_performance,
                'last_analysis': self._last_analysis
            }
            joblib.dump(analyzer_data, 'fire_incident_analyzer.pkl')
            print("Analyzer saved successfully")
        except Exception as e:
            print(f"Error saving analyzer: {e}")
    def compare_incident(self, incident_data, historical_data):
        """Compare new incident with historical data with robust error handling"""
        try:
            if not historical_data or len(historical_data) < 3:
                return {
                    'error': 'Insufficient historical data for comparison',
                    'similar_average': 8.5,
                    'time_difference': 0,
                    'performance_ratio': 1.0,
                    'similar_count': 0
                }
        
            df_historical = pd.DataFrame(historical_data)
        
            # Try to predict, but handle failures gracefully
            prediction_result = self.predict_response_time(incident_data)
            category_result = self.predict_response_category(incident_data)
        
            # Calculate historical averages as fallback
            avg_response_time = df_historical['response_time'].mean()
            current_time = incident_data.get('response_time', prediction_result.get('predicted_response_time', avg_response_time))
        
            # Find similar incidents using simple distance-based matching
            similar_incidents = self.find_similar_incidents_simple(df_historical, incident_data)
        
            if len(similar_incidents) > 0:
                similar_avg = similar_incidents['response_time'].mean()
                similar_count = len(similar_incidents)
            else:
                similar_avg = avg_response_time
                similar_count = len(df_historical)
        
            # Performance comparison
            time_diff = current_time - similar_avg
            performance_ratio = current_time / similar_avg if similar_avg > 0 else 1.0
        
            comparison = {
                'historical_average': float(avg_response_time),
                'similar_average': float(similar_avg),
                'time_difference': float(time_diff),
                'performance_ratio': float(performance_ratio),
                'similar_count': int(similar_count),
                'prediction': prediction_result,
                'predicted_category': category_result,
                'expected_time': float(similar_avg),
                'actual_time': float(current_time),
                'comparison_method': 'distance_based' if len(similar_incidents) > 0 else 'historical_average'
            }
        
            # Add warning if prediction had issues
            if prediction_result.get('warning'):
                comparison['prediction_warning'] = prediction_result['warning']
            if category_result.get('warning'):
                comparison['category_warning'] = category_result['warning']
        
            self._last_analysis = {
                'incident': incident_data,
                'comparison': comparison,
                'timestamp': datetime.now().isoformat()
            }
        
            return comparison
        
        except Exception as e:
            print(f"Comparison error: {e}")
            return {
                'error': str(e),
                'similar_average': 8.5,
                'time_difference': 0,
                'performance_ratio': 1.0,
                'similar_count': 0
            }
    def find_similar_incidents_simple(self, df, incident):
        """Find similar incidents using simple distance-based matching"""
        try:
            current_distance = incident.get('distance', 0)
            current_type = incident.get('type', incident.get('type_of_occupancy', 'Other'))
        
            if current_distance == 0:
                return pd.DataFrame()
        
            # Find incidents within 25% distance range and same type
            distance_range = 0.25 # 25% range
            lower_bound = current_distance * (1 - distance_range)
            upper_bound = current_distance * (1 + distance_range)
        
            similar = df[
                (df['distance'] >= lower_bound) &
                (df['distance'] <= upper_bound)
            ].copy()
        
            # If we have type information, filter by type
            if 'type' in df.columns and current_type:
                similar = similar[similar['type'] == current_type]
        
            return similar
        
        except Exception as e:
            print(f"Simple similarity error: {e}")
            return pd.DataFrame()
    def predict_response_time(self, incident_data):
        """Predict response time using regression model with consistent features"""
        try:
            if not self.regressor:
                return {'predicted_response_time': 8.5, 'confidence': 0.5, 'unit': 'minutes'}
         
            # Prepare features - ensure all required features are present with proper defaults
            features_df = pd.DataFrame([incident_data])
         
            # Add missing fields with defaults if they don't exist
            required_fields = {
                'severity': 'moderate',
                'weather': incident_data.get('weather_condition', 'Sunny'),
                'temperature': incident_data.get('temperature_c', 25),
                'humidity': incident_data.get('humidity_pct', 70),
                'wind_speed': incident_data.get('wind_speed_kmh', 10),
                'distance': incident_data.get('distance', 3.0),
                'response_time': incident_data.get('response_time_min', 5),
                'timestamp': incident_data.get('timestamp', datetime.now().isoformat()),
                'type': incident_data.get('type_of_occupancy', 'Other')
            }
         
            for field, default_value in required_fields.items():
                if field not in features_df.columns:
                    features_df[field] = default_value
         
            # Ensure we have the exact features the model was trained on
            features_processed = self.preprocess_data(features_df, for_classification=False)
         
            # DEBUG: Check what features we have vs what model expects
            print(f"Processed features: {list(features_processed.columns)}")
            print(f"Model expects: {self.regression_features}")
         
            # Ensure we only use the features the model was trained on
            available_features = [f for f in self.regression_features if f in features_processed.columns]
         
            if len(available_features) != len(self.regression_features):
                print(f"Warning: Feature mismatch. Available: {available_features}, Expected: {self.regression_features}")
                # Use fallback prediction
                return {
                    'predicted_response_time': 8.5,
                    'confidence': 0.3,
                    'unit': 'minutes',
                    'warning': 'Feature mismatch - using fallback prediction'
                }
         
            prediction = self.regressor.predict(features_processed[available_features])[0]
            confidence = 0.8 # Default confidence
         
            return {
                'predicted_response_time': float(prediction),
                'confidence': float(confidence),
                'unit': 'minutes'
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return {'predicted_response_time': 8.5, 'confidence': 0.5, 'unit': 'minutes'}
    def predict_response_category(self, incident_data):
        """Predict response category using classification model with consistent features"""
        try:
            if not self.classifier:
                return {
                    'predicted_category': 'moderate',
                    'display_category': 'Standard',
                    'confidence': 0.5
                }
         
            # Prepare features - ensure all required features are present
            features_df = pd.DataFrame([incident_data])
         
            # Add missing fields with defaults if they don't exist
            required_fields = {
                'severity': 'moderate',
                'weather': incident_data.get('weather_condition', 'Sunny'),
                'temperature': incident_data.get('temperature_c', 25),
                'humidity': incident_data.get('humidity_pct', 70),
                'wind_speed': incident_data.get('wind_speed_kmh', 10),
                'distance': incident_data.get('distance', 3.0),
                'response_time': incident_data.get('response_time_min', 5),
                'timestamp': incident_data.get('timestamp', datetime.now().isoformat()),
                'type': incident_data.get('type_of_occupancy', 'Other')
            }
         
            for field, default_value in required_fields.items():
                if field not in features_df.columns:
                    features_df[field] = default_value
         
            # Add response category for processing
            features_df['response_category'] = self.create_response_categories([features_df['response_time'].iloc[0]])[0]
         
            features_processed = self.preprocess_data(features_df, for_classification=True)
         
            # DEBUG: Check what features we have vs what model expects
            print(f"Classification - Processed features: {list(features_processed.columns)}")
            print(f"Classification - Model expects: {self.classification_features}")
         
            # Ensure we only use the features the model was trained on
            available_features = [f for f in self.classification_features if f in features_processed.columns]
         
            if len(available_features) != len(self.classification_features):
                print(f"Warning: Classification feature mismatch. Available: {available_features}, Expected: {self.classification_features}")
                return {
                    'predicted_category': 'moderate',
                    'display_category': 'Standard',
                    'confidence': 0.3,
                    'warning': 'Feature mismatch - using fallback prediction'
                }
         
            prediction = self.classifier.predict(features_processed[available_features])[0]
            probabilities = self.classifier.predict_proba(features_processed[available_features])[0]
            confidence = float(np.max(probabilities))
         
            category_map = {
                'very_fast': 'Excellent',
                'fast': 'Good',
                'moderate': 'Standard',
                'slow': 'Needs Review',
                'very_slow': 'Critical Delay'
            }
         
            return {
                'predicted_category': prediction,
                'display_category': category_map.get(prediction, 'Standard'),
                'confidence': confidence
            }
        except Exception as e:
            print(f"Category prediction error: {e}")
            return {
                'predicted_category': 'moderate',
                'display_category': 'Standard',
                'confidence': 0.5
            }
# Initialize global analyzer
fire_analyzer = FireIncidentAnalyzer()
# Load initial data
load_incidents_from_csv()
# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'incidents_loaded': len(fire_incidents),
        'analyzer_ready': fire_analyzer is not None
    })
@app.route('/api/model-status', methods=['GET'])
def model_status():
    """Get current model status and accuracy"""
    try:
        if fire_analyzer and fire_analyzer._baseline_performance:
            performance = fire_analyzer._baseline_performance
            status = {
                'status': 'ready',
                'accuracy': performance['classification']['accuracy'],
                'r2_score': performance['regression']['r2'],
                'training_samples': performance['training_samples'],
                'last_trained': performance['timestamp'],
                'feature_importance': performance['feature_importance']
            }
        else:
            status = {
                'status': 'needs_training',
                'accuracy': 0.0,
                'r2_score': 0.0,
                'training_samples': 0,
                'message': 'No trained model available'
            }
  
        return jsonify(status)
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)}), 500
@app.route('/api/analyzer-status', methods=['GET'])
def analyzer_status():
    """Get analyzer status for compatibility"""
    return model_status()
@app.route('/api/compare', methods=['POST'])
def compare_incident():
    """Compare incident performance"""
    try:
        data = request.json
        if not fire_analyzer:
            return jsonify({'error': 'Analyzer not ready'}), 500
  
        comparison = fire_analyzer.compare_incident(data, fire_incidents)
        return jsonify(comparison)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/performance-trends', methods=['GET'])
def performance_trends():
    """Get performance trends over current incidents"""
    try:
        # Align minimum data requirement with frontend (needs >=5 incidents)
        if len(fire_incidents) < 5:
            return jsonify({
                'error': 'Need at least 5 incidents for reliable training',
                'current_count': len(fire_incidents),
                'suggestion': 'Continue collecting data before training'
            }), 400
  
        performance = fire_analyzer.train_analyzer(fire_incidents)
  
        if 'error' in performance:
            return jsonify(performance), 500
  
        # ADD OVERFITTING CHECK
        overfitting_warning = None
        if fire_analyzer._baseline_performance:
            reg_r2 = fire_analyzer._baseline_performance['regression']['r2']
            clf_accuracy = fire_analyzer._baseline_performance['classification']['accuracy']
      
            if reg_r2 > 0.95: # Suspiciously high R²
                overfitting_warning = "High R² detected - model may be overfitting"
            elif clf_accuracy > 0.95: # Suspiciously high accuracy
                overfitting_warning = "High accuracy detected - model may be overfitting"
  
        return jsonify({
            'message': 'Training completed successfully',
            'performance': performance,
            'overfitting_warning': overfitting_warning,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/train', methods=['POST'])
def train_analyzer():
    """Train analyzer with current incidents"""
    try:
        # Align minimum data requirement with frontend (needs >=5 incidents)
        if len(fire_incidents) < 5:
            return jsonify({
                'error': 'Need at least 5 incidents for reliable training',
                'current_count': len(fire_incidents),
                'suggestion': 'Continue collecting data before training'
            }), 400
  
        performance = fire_analyzer.train_analyzer(fire_incidents)
  
        if 'error' in performance:
            return jsonify(performance), 500
  
        # ADD OVERFITTING CHECK
        overfitting_warning = None
        if fire_analyzer._baseline_performance:
            reg_r2 = fire_analyzer._baseline_performance['regression']['r2']
            clf_accuracy = fire_analyzer._baseline_performance['classification']['accuracy']
      
            if reg_r2 > 0.95: # Suspiciously high R²
                overfitting_warning = "High R² detected - model may be overfitting"
            elif clf_accuracy > 0.95: # Suspiciously high accuracy
                overfitting_warning = "High accuracy detected - model may be overfitting"
  
        return jsonify({
            'message': 'Training completed successfully',
            'performance': performance,
            'overfitting_warning': overfitting_warning,
            'timestamp': datetime.now().isoformat()
        })
  
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    """Get all incidents"""
    try:
        return jsonify({
            'incidents': fire_incidents,
            'count': len(fire_incidents),
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/incidents', methods=['POST'])
def store_incident():
    """Store new incident with complete CSV data and robust error handling"""
    try:
        data = request.json
        print(f"🔍 DEBUG - Received data: {data}")
   
        # Validate required fields for detailed reports
        required_fields = ['location', 'type_of_occupancy']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({'error': f'Missing required field: {field}'}), 400
   
        # Set default values for missing fields
        defaults = {
            'station': 'Santa Cruz, Laguna',
            'date_of_response': datetime.now().strftime('%Y-%m-%d'),
            'responding_unit': 'Shift B',
            'time_received': '08:00',
            'time_dispatched': '08:00',
            'time_arrival': '08:05',
            'response_time_min': 5,
            'distance': 3.3,
            'alarm_status': 'FA',
            'injured_civ': 0,
            'injured_bfp': 0,
            'death_civ': 0,
            'death_bfp': 0,
            'remarks': 'Case Closed',
            'temperature_c': 25.0,
            'humidity_pct': 70,
            'wind_speed_kmh': 10.0,
            'precipitation_mm': 0.0,
            'weather_condition': 'Sunny',
            'road_condition': 'Dry'
        }
   
        for key, default_value in defaults.items():
            if key not in data:
                data[key] = default_value
   
        # Ensure numeric fields are properly typed
        numeric_fields = {
            'response_time_min': int, 'distance': float, 'injured_civ': int,
            'injured_bfp': int, 'death_civ': int, 'death_bfp': int,
            'temperature_c': float, 'humidity_pct': int, 'wind_speed_kmh': float,
            'precipitation_mm': float
        }
   
        for field, converter in numeric_fields.items():
            if field in data:
                try:
                    data[field] = converter(data[field])
                except (ValueError, TypeError):
                    data[field] = defaults.get(field, 0)
   
        # Create incident ID and timestamp
        incident_id = len(fire_incidents) + 1
        data['id'] = incident_id
        data['timestamp'] = datetime.now().isoformat()
   
        # Add to memory storage
        fire_incidents.append(data)
        print(f"✅ DEBUG - Added to memory. Total incidents: {len(fire_incidents)}")
   
        # Save to CSV
        success = append_to_csv(data)
        print(f"✅ DEBUG - CSV save status: {success}")
   
        if not success:
            return jsonify({'error': 'Failed to save to CSV'}), 500
   
        # Perform analysis if analyzer is ready - WITH ERROR HANDLING
        comparison_result = {}
        if fire_analyzer:
            try:
                # Create ML-compatible incident data
                ml_incident_data = {
                    'severity': determine_severity(data),
                    'type': data.get('type_of_occupancy', 'Other'),
                    'temperature': data.get('temperature_c', 25),
                    'humidity': data.get('humidity_pct', 70),
                    'wind_speed': data.get('wind_speed_kmh', 10),
                    'response_time': data.get('response_time_min', 5),
                    'distance': data.get('distance', 3.0),
                    'weather': data.get('weather_condition', 'Sunny'),
                    'timestamp': data['timestamp'],
                    'road_condition': data.get('road_condition', 'Dry')
                }
                print(f"🔍 DEBUG - ML incident data: {ml_incident_data}")
           
                recent_historical = fire_incidents[-30:] if len(fire_incidents) > 30 else fire_incidents
                comparison_result = fire_analyzer.compare_incident(ml_incident_data, recent_historical)
                print(f"✅ DEBUG - Comparison result: {comparison_result}")
              
            except Exception as e:
                print(f"❌ DEBUG - Analysis failed: {e}")
                comparison_result = {
                    'error': 'Analysis failed',
                    'similar_average': 8.5,
                    'time_difference': 0,
                    'performance_ratio': 1.0
                }
        else:
            print("❌ DEBUG - Fire analyzer not available")
   
        response_data = {
            'message': 'Incident stored successfully',
            'id': incident_id,
            'timestamp': data['timestamp'],
            'csv_saved': success,
            'comparison_analysis': comparison_result
        }
      
        print(f"✅ DEBUG - Sending response: {response_data}")
        return jsonify(response_data)
   
    except Exception as e:
        print(f"❌ DEBUG - Error storing incident: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get incident statistics"""
    try:
        if not fire_incidents:
            return jsonify({'statistics': {}, 'message': 'No incidents available'})
  
        df = pd.DataFrame(fire_incidents)
  
        stats = {
            'total_incidents': len(df),
            'average_response_time': float(df['response_time'].mean()),
            'average_distance': float(df['distance'].mean()),
            'by_severity': df['severity'].value_counts().to_dict() if 'severity' in df.columns else {},
            'by_type': df['type'].value_counts().to_dict() if 'type' in df.columns else {},
            'monthly_trends': df.groupby(df['timestamp'].dt.to_period('M'))['response_time'].mean().to_dict() if 'timestamp' in df.columns else {}
        }
  
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/load-csv', methods=['POST'])
def load_csv():
    """Load incidents from CSV file"""
    try:
        load_incidents_from_csv()
        return jsonify({
            'message': 'CSV loaded successfully',
            'incidents_loaded': len(fire_incidents)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/predict-category', methods=['POST'])
def predict_category():
    """Predict response category for incident"""
    try:
        data = request.json
        if not fire_analyzer:
            return jsonify({'error': 'Analyzer not ready'}), 500
  
        category = fire_analyzer.predict_response_category(data)
        return jsonify(category)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/improvement-suggestions', methods=['POST'])
def improvement_suggestions():
    """Get improvement suggestions based on incident data"""
    try:
        incident_data = request.json
        if not fire_incidents:
            return jsonify({
                'suggestions': [get_initial_training_suggestion()],
                'analysis': {},
                'error': 'No historical data available'
            })
  
        # Perform comparison analysis
        comparison = fire_analyzer.compare_incident(incident_data, fire_incidents) if fire_analyzer else {}
  
        # Generate distance-based suggestions
        suggestions = generate_distance_based_suggestions(incident_data, comparison)
  
        return jsonify({
            'suggestions': suggestions,
            'analysis': {
                'performance_ratio': comparison.get('performance_ratio', 1.0),
                'time_difference': comparison.get('time_difference', 0),
                'expected_time': comparison.get('expected_response_time', 0),
                'actual_time': comparison.get('actual_response_time', 0)
            },
            'timestamp': datetime.now().isoformat()
        })
  
    except Exception as e:
        return jsonify({'error': str(e)}), 500
def generate_distance_based_suggestions(incident_data, comparison):
    """Generate suggestions based on similar distance/destination incidents"""
    suggestions = []
    if not fire_incidents or len(fire_incidents) < 5:
        return [get_initial_training_suggestion()]
    df = pd.DataFrame(fire_incidents)
    # Get similar incidents based on distance and location patterns
    similar_incidents = find_similar_distance_incidents(incident_data, df)
    similar_destination_incidents = find_similar_destination_incidents(incident_data, df)
    # Generate distance-based suggestions
    suggestions.extend(generate_distance_comparison_suggestions(incident_data, similar_incidents, comparison))
    # Generate destination-based suggestions
    suggestions.extend(generate_destination_comparison_suggestions(incident_data, similar_destination_incidents, comparison))
    # Generate route optimization suggestions
    suggestions.extend(generate_route_optimization_suggestions(incident_data, df, comparison))
    # Limit to most relevant suggestions
    return sorted(suggestions, key=lambda x: x.get('priority_score', 0), reverse=True)[:3]
def find_similar_distance_incidents(incident_data, df):
    """Find incidents with similar distances"""
    try:
        current_distance = incident_data.get('distance', 0)
        if current_distance == 0:
            return pd.DataFrame()
  
        # Find incidents within 20% distance range
        distance_range = 0.2 # 20% range
        lower_bound = current_distance * (1 - distance_range)
        upper_bound = current_distance * (1 + distance_range)
  
        similar = df[
            (df['distance'] >= lower_bound) &
            (df['distance'] <= upper_bound)
        ].copy()
  
        return similar
  
    except Exception as e:
        print(f"Error finding similar distance incidents: {e}")
        return pd.DataFrame()
def find_similar_destination_incidents(incident_data, df):
    """Find incidents with similar destination patterns"""
    try:
        similar = df.copy()
  
        # Filter by location pattern (if available)
        current_location = incident_data.get('location', '').lower()
        if current_location and 'location' in df.columns:
            # Find incidents with similar location keywords
            location_keywords = extract_location_keywords(current_location)
            if location_keywords:
                location_mask = df['location'].str.lower().apply(
                    lambda x: any(keyword in str(x) for keyword in location_keywords)
                )
                similar = similar[location_mask]
  
        # Filter by incident type
        current_type = incident_data.get('type', '')
        if current_type and 'type' in df.columns:
            similar = similar[similar['type'] == current_type]
  
        return similar
  
    except Exception as e:
        print(f"Error finding similar destination incidents: {e}")
        return pd.DataFrame()
def extract_location_keywords(location):
    """Extract relevant location keywords for matching"""
    common_areas = ['santa cruz', 'laguna', 'barangay', 'poblacion', 'highway', 'road', 'street']
    keywords = []
    for area in common_areas:
        if area in location.lower():
            keywords.append(area)
    return keywords if keywords else ['santa cruz'] # Default to municipality
def generate_distance_comparison_suggestions(incident_data, similar_incidents, comparison):
    """Generate suggestions based on distance comparisons"""
    suggestions = []
    try:
        if len(similar_incidents) < 3:
            return suggestions
  
        current_distance = incident_data.get('distance', 0)
        current_response = incident_data.get('response_time', 0)
  
        # Calculate average performance for similar distances
        avg_response_similar = similar_incidents['response_time'].mean()
        avg_response_per_km = similar_incidents['response_time'].mean() / similar_incidents['distance'].mean()
  
        current_performance_ratio = current_response / current_distance if current_distance > 0 else 0
        avg_performance_ratio = avg_response_per_km
  
        performance_difference = current_performance_ratio - avg_performance_ratio
        response_difference = current_response - avg_response_similar
  
        # Generate suggestion based on performance difference
        if response_difference > 2: # More than 2 minutes slower
            suggestions.append({
                'type': 'distance_performance',
                'priority': 'high',
                'priority_score': 90,
                'title': '🚨 Slower Than Similar Distance Responses',
                'description': f'Response time of {current_response}min for {current_distance}km is {response_difference:.1f}min slower than average for similar distances ({avg_response_similar:.1f}min).',
                'actions': [
                    'Review route selection for this distance range',
                    'Check for consistent traffic patterns at this distance',
                    'Consider pre-positioning for frequent medium-distance calls'
                ],
                'impact': f'Could reduce response time by {response_difference:.1f} minutes',
                'data_source': f'Based on {len(similar_incidents)} incidents with similar distance ({current_distance}km)',
                'metrics': {
                    'current_time': current_response,
                    'average_time': avg_response_similar,
                    'time_difference': response_difference,
                    'comparison_count': len(similar_incidents)
                }
            })
  
        elif response_difference < -1: # More than 1 minute faster
            suggestions.append({
                'type': 'distance_success',
                'priority': 'low',
                'priority_score': 40,
                'title': '✅ Excellent Distance Performance',
                'description': f'Response time of {current_response}min for {current_distance}km is {abs(response_difference):.1f}min faster than average for similar distances.',
                'actions': [
                    'Document successful strategies for this distance range',
                    'Share best practices with other response teams',
                    'Maintain current deployment patterns'
                ],
                'impact': 'Maintain superior performance for this distance',
                'data_source': f'Based on {len(similar_incidents)} incidents with similar distance',
                'metrics': {
                    'current_time': current_response,
                    'average_time': avg_response_similar,
                    'time_difference': response_difference,
                    'comparison_count': len(similar_incidents)
                }
            })
  
        # Check performance ratio (time per km)
        if performance_difference > 1.0: # Significantly higher minutes per km
            suggestions.append({
                'type': 'efficiency',
                'priority': 'medium',
                'priority_score': 70,
                'title': '📊 Response Efficiency Opportunity',
                'description': f'Current rate of {current_performance_ratio:.1f} min/km is higher than average {avg_performance_ratio:.1f} min/km for similar distances.',
                'actions': [
                    'Analyze travel speed patterns for this distance category',
                    'Review vehicle performance and maintenance schedules',
                    'Consider alternative response modes for medium distances'
                ],
                'impact': f'Improve efficiency by {performance_difference:.1f} min/km',
                'data_source': f'Comparison of {len(similar_incidents)} similar distance responses',
                'metrics': {
                    'current_efficiency': current_performance_ratio,
                    'average_efficiency': avg_performance_ratio,
                    'efficiency_difference': performance_difference
                }
            })
      
    except Exception as e:
        print(f"Error generating distance comparison suggestions: {e}")
    return suggestions
def generate_destination_comparison_suggestions(incident_data, similar_incidents, comparison):
    """Generate suggestions based on destination area comparisons"""
    suggestions = []
    try:
        if len(similar_incidents) < 3:
            return suggestions
  
        current_response = incident_data.get('response_time', 0)
        current_location = incident_data.get('location', 'Unknown')
  
        # Calculate average for similar destinations
        avg_destination_response = similar_incidents['response_time'].mean()
        response_difference = current_response - avg_destination_response
  
        # Get most common issues for this destination area
        common_issues = analyze_destination_issues(similar_incidents)
  
        if response_difference > 3: # More than 3 minutes slower
            suggestion = {
                'type': 'destination_performance',
                'priority': 'high',
                'priority_score': 85,
                'title': '📍 Slower Response for This Area',
                'description': f'Response time to {current_location} was {current_response}min, {response_difference:.1f}min slower than area average.',
                'actions': [
                    'Review traffic patterns and road conditions in this area',
                    'Consider pre-positioning during high-risk periods',
                    'Coordinate with local authorities for route optimization'
                ],
                'impact': f'Potential {response_difference:.1f} minute improvement',
                'data_source': f'Based on {len(similar_incidents)} responses to similar areas',
                'metrics': {
                    'current_time': current_response,
                    'area_average': avg_destination_response,
                    'time_difference': response_difference
                }
            }
      
            # Add area-specific issues if found
            if common_issues:
                suggestion['actions'].extend(common_issues)
                suggestion['description'] += f' Common issues: {", ".join(common_issues[:2])}.'
      
            suggestions.append(suggestion)
  
        elif response_difference < -2: # More than 2 minutes faster
            suggestions.append({
                'type': 'destination_success',
                'priority': 'low',
                'priority_score': 30,
                'title': '✅ Excellent Area Response',
                'description': f'Response to {current_location} was {abs(response_difference):.1f}min faster than area average.',
                'actions': [
                    'Document successful response strategies for this area',
                    'Share area-specific best practices',
                    'Maintain current response protocols'
                ],
                'impact': 'Maintain area response excellence',
                'data_source': f'Based on {len(similar_incidents)} area responses',
                'metrics': {
                    'current_time': current_response,
                    'area_average': avg_destination_response,
                    'time_difference': response_difference
                }
            })
      
    except Exception as e:
        print(f"Error generating destination comparison suggestions: {e}")
    return suggestions
def analyze_destination_issues(similar_incidents):
    """Analyze common issues for a destination area"""
    issues = []
    try:
        # Check for weather patterns
        if 'weather' in similar_incidents.columns:
            weather_impact = similar_incidents.groupby('weather')['response_time'].mean()
            adverse_weather = weather_impact[weather_impact > weather_impact.median() * 1.2]
            if len(adverse_weather) > 0:
                issues.append(f"Watch for {list(adverse_weather.index)[0]} conditions")
  
        # Check for time patterns
        if 'timestamp' in similar_incidents.columns:
            similar_incidents['timestamp'] = pd.to_datetime(similar_incidents['timestamp'])
            similar_incidents['hour'] = similar_incidents['timestamp'].dt.hour
            hourly_performance = similar_incidents.groupby('hour')['response_time'].mean()
            peak_hours = hourly_performance[hourly_performance > hourly_performance.median() * 1.15]
            if len(peak_hours) > 0:
                issues.append(f"Avoid peak hours: {list(peak_hours.index)}")
          
    except Exception as e:
        print(f"Error analyzing destination issues: {e}")
    return issues[:2] # Return top 2 issues
def generate_route_optimization_suggestions(incident_data, df, comparison):
    """Generate route-specific optimization suggestions"""
    suggestions = []
    try:
        current_response = incident_data.get('response_time', 0)
        expected_time = comparison.get('expected_response_time', current_response)
        time_difference = comparison.get('time_difference', 0)
  
        if time_difference > 5: # More than 5 minutes slower than expected
            suggestions.append({
                'type': 'route_optimization',
                'priority': 'high',
                'priority_score': 95,
                'title': '🛣️ Significant Route Delay Detected',
                'description': f'Actual response time ({current_response}min) was {time_difference:.1f}min slower than expected ({expected_time:.1f}min).',
                'actions': [
                    'Review real-time traffic and road conditions',
                    'Consider alternative routes for future responses',
                    'Check for road construction or closures',
                    'Verify navigation system accuracy'
                ],
                'impact': f'Potential {time_difference:.1f} minute improvement with better routing',
                'data_source': 'Machine learning route prediction vs actual performance',
                'metrics': {
                    'actual_time': current_response,
                    'expected_time': expected_time,
                    'delay': time_difference
                }
            })
  
        # Check for consistent delays in similar routes
        similar_routes = find_similar_route_patterns(incident_data, df)
        if len(similar_routes) > 5:
            avg_route_delay = similar_routes['response_time'].mean() - similar_routes['distance'].mean() * 3 # Base expectation
            if avg_route_delay > 3:
                suggestions.append({
                    'type': 'route_pattern',
                    'priority': 'medium',
                    'priority_score': 65,
                    'title': '📈 Consistent Route Challenges',
                    'description': f'This route pattern shows consistent delays averaging {avg_route_delay:.1f}min above expectations.',
                    'actions': [
                        'Analyze traffic flow patterns for this route category',
                        'Develop pre-planned alternative routes',
                        'Coordinate with traffic management for priority access'
                    ],
                    'impact': f'Address consistent {avg_route_delay:.1f} minute delays',
                    'data_source': f'Pattern analysis of {len(similar_routes)} similar routes',
                    'metrics': {
                        'average_delay': avg_route_delay,
                        'pattern_count': len(similar_routes)
                    }
                })
          
    except Exception as e:
        print(f"Error generating route optimization suggestions: {e}")
    return suggestions
def find_similar_route_patterns(incident_data, df):
    """Find incidents with similar route characteristics"""
    try:
        similar = df.copy()
  
        # Filter by distance range
        current_distance = incident_data.get('distance', 0)
        if current_distance > 0:
            similar = similar[
                (similar['distance'] >= current_distance * 0.7) &
                (similar['distance'] <= current_distance * 1.3)
            ]
  
        # Filter by time of day if available
        if 'timestamp' in incident_data and 'timestamp' in df.columns:
            incident_hour = pd.to_datetime(incident_data['timestamp']).hour
            similar['hour'] = pd.to_datetime(similar['timestamp']).dt.hour
            similar = similar[similar['hour'].between(incident_hour - 2, incident_hour + 2)]
  
        return similar
  
    except Exception as e:
        print(f"Error finding similar route patterns: {e}")
        return pd.DataFrame()
def get_initial_training_suggestion():
    """Suggestion when not enough data"""
    return {
        'type': 'training',
        'priority': 'medium',
        'priority_score': 50,
        'title': '📈 Collect More Distance Data',
        'description': 'Need more incident data to provide distance-based comparisons:',
        'actions': [
            'Record response distances accurately',
            'Include destination location details',
            'Track route variations and conditions'
        ],
        'impact': 'Enable distance-based performance analysis',
        'data_source': 'System recommendation',
        'metrics': {}
    }
def generate_performance_feedback(incident_data, comparison, historical_data):
    """Generate comprehensive performance feedback"""
    # Basic performance metrics
    current_time = incident_data.get('response_time', 0)
    expected_time = comparison.get('similar_average', current_time)
    performance_ratio = comparison.get('performance_ratio', 1.0)
    # Determine performance category
    if performance_ratio < 0.9:
        performance_category = 'excellent'
        performance_message = 'Excellent response time!'
    elif performance_ratio < 1.1:
        performance_category = 'good'
        performance_message = 'Good response time'
    elif performance_ratio < 1.3:
        performance_category = 'average'
        performance_message = 'Average response time - room for improvement'
    else:
        performance_category = 'needs_improvement'
        performance_message = 'Response time needs improvement'
    # Generate detailed feedback
    feedback = {
        'performance_category': performance_category,
        'performance_message': performance_message,
        'metrics': {
            'current_response_time': current_time,
            'expected_response_time': round(expected_time, 1),
            'performance_ratio': round(performance_ratio, 2),
            'time_difference': round(comparison.get('time_difference', 0), 1),
            'similar_incidents_count': comparison.get('similar_count', 0)
        },
        'comparison_insights': generate_comparison_insights(incident_data, comparison, historical_data),
        'improvement_suggestions': generate_improvement_suggestions(incident_data, comparison, historical_data),
        'training_progress': {
            'total_incidents': len(historical_data),
            'model_ready': len(historical_data) >= 10,
            'recommended_training': len(historical_data) >= 5
        }
    }
    return feedback
def generate_comparison_insights(incident_data, comparison, historical_data):
    """Generate insights by comparing with historical data"""
    insights = []
    current_time = incident_data.get('response_time', 0)
    expected_time = comparison.get('similar_average', current_time)
    time_diff = comparison.get('time_difference', 0)
    # Time-based insights
    if time_diff < -2:
        insights.append({
            'type': 'positive',
            'title': '🚀 Faster Than Expected',
            'description': f'Your response was {abs(time_diff):.1f} minutes faster than similar historical incidents.'
        })
    elif time_diff > 2:
        insights.append({
            'type': 'improvement',
            'title': '⏱️ Slower Response Detected',
            'description': f'Response was {time_diff:.1f} minutes slower than similar incidents.'
        })
    # Distance efficiency insight
    distance = incident_data.get('distance', 0)
    if distance > 0 and current_time > 0:
        efficiency_ratio = current_time / distance
        avg_efficiency = expected_time / distance if distance > 0 else 0
 
        if efficiency_ratio > avg_efficiency * 1.2:
            insights.append({
                'type': 'improvement',
                'title': '📏 Distance Efficiency',
                'description': f'Time per km ({efficiency_ratio:.1f} min/km) is higher than average.'
            })
    # Weather impact insight
    weather = incident_data.get('weather', '').lower()
    adverse_weather = ['rain', 'storm', 'typhoon', 'heavy']
    if any(adv in weather for adv in adverse_weather):
        insights.append({
            'type': 'context',
            'title': '🌧️ Weather Conditions',
            'description': 'Adverse weather conditions may have impacted response time.'
        })
    return insights
def generate_improvement_suggestions(incident_data, comparison, historical_data):
    """Generate actionable improvement suggestions"""
    suggestions = []
    current_time = incident_data.get('response_time', 0)
    expected_time = comparison.get('similar_average', current_time)
    time_diff = comparison.get('time_difference', 0)
    # Response time suggestions
    if time_diff > 5:
        suggestions.append({
            'type': 'response_time',
            'priority': 'high',
            'title': '🚨 Significant Delay',
            'description': f'Response time was {time_diff:.1f} minutes slower than expected.',
            'actions': [
                'Review dispatch procedures',
                'Check vehicle readiness',
                'Analyze route selection'
            ],
            'expected_improvement': f'Reduce by {min(time_diff, 10):.1f} minutes'
        })
    elif time_diff > 2:
        suggestions.append({
            'type': 'response_time',
            'priority': 'medium',
            'title': '📊 Moderate Delay',
            'description': f'Response time was {time_diff:.1f} minutes slower than average.',
            'actions': [
                'Optimize route planning',
                'Review traffic patterns',
                'Pre-position for common locations'
            ],
            'expected_improvement': f'Reduce by {time_diff:.1f} minutes'
        })
    # Distance-based suggestions
    distance = incident_data.get('distance', 0)
    if distance > 10 and current_time/distance > 3:
        suggestions.append({
            'type': 'long_distance',
            'priority': 'medium',
            'title': '📍 Long Distance Response',
            'description': f'Long distance response ({distance}km) shows efficiency opportunities.',
            'actions': [
                'Consider satellite station deployment',
                'Optimize highway response protocols',
                'Review long-distance equipment'
            ],
            'expected_improvement': 'Improve long-distance efficiency'
        })
    # Training suggestions
    if len(historical_data) < 10:
        suggestions.append({
            'type': 'training',
            'priority': 'low',
            'title': '📈 More Data Needed',
            'description': f'Only {len(historical_data)} incidents recorded. More data improves accuracy.',
            'actions': [
                'Continue recording incidents',
                'Include varied scenarios',
                'Track all response metrics'
            ],
            'expected_improvement': 'Better performance predictions'
        })
    return suggestions
def generate_training_recommendations(performance):
    """Generate recommendations for model improvement"""
    recommendations = []
    accuracy = performance['classification']['accuracy']
    r2_score = performance['regression']['r2']
    if accuracy < 0.7:
        recommendations.append({
            'type': 'accuracy',
            'priority': 'high',
            'suggestion': 'Collect more diverse incident data',
            'reason': f'Current accuracy ({accuracy:.1%}) below optimal level'
        })
    if r2_score < 0.5:
        recommendations.append({
            'type': 'prediction',
            'priority': 'medium',
            'suggestion': 'Include more weather and traffic data',
            'reason': f'Prediction quality (R²={r2_score:.2f}) can be improved'
        })
    if performance['training_samples'] < 20:
        recommendations.append({
            'type': 'data_volume',
            'priority': 'medium',
            'suggestion': 'Record more incidents for better patterns',
            'reason': f'Only {performance["training_samples"]} incidents available'
        })
    return recommendations
@app.route('/api/performance-feedback', methods=['POST'])
def get_performance_feedback():
    """Get detailed performance feedback for an incident"""
    try:
        data = request.json
        incident_data = data.get('incident_data', {})
 
        if not fire_incidents:
            return jsonify({
                'status': 'no_data',
                'message': 'Not enough historical data for comparison',
                'suggestions': [{
                    'type': 'data_collection',
                    'priority': 'medium',
                    'title': '📊 Collect More Data',
                    'description': 'Continue recording incidents to build performance insights.',
                    'actions': [
                        'Record detailed incident information',
                        'Include accurate response times and distances',
                        'Track weather and road conditions'
                    ],
                    'impact': 'Enable detailed performance analysis'
                }]
            })
 
        # Get comparison analysis
        comparison = fire_analyzer.compare_incident(incident_data, fire_incidents)
 
        # Generate performance feedback
        feedback = generate_performance_feedback(incident_data, comparison, fire_incidents)
 
        return jsonify(feedback)
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/training-feedback', methods=['GET'])
def get_training_feedback():
    """Get feedback about model training status and performance"""
    try:
        if not fire_analyzer._baseline_performance:
            return jsonify({
                'status': 'not_trained',
                'message': 'Model needs training',
                'recommendation': 'Train model with at least 5 incidents'
            })
 
        performance = fire_analyzer._baseline_performance
        accuracy = performance['classification']['accuracy']
        r2_score = performance['regression']['r2']
 
        # Determine model quality
        if accuracy > 0.8 and r2_score > 0.6:
            status = 'excellent'
            message = 'Model performing excellently'
        elif accuracy > 0.7 and r2_score > 0.5:
            status = 'good'
            message = 'Model performing well'
        elif accuracy > 0.6:
            status = 'fair'
            message = 'Model performance is fair'
        else:
            status = 'needs_improvement'
            message = 'Model needs more training data'
 
        return jsonify({
            'status': status,
            'message': message,
            'metrics': {
                'accuracy': round(accuracy, 3),
                'r2_score': round(r2_score, 3),
                'training_samples': performance['training_samples']
            },
            'recommendations': generate_training_recommendations(performance)
        })
 
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/comprehensive-feedback', methods=['GET', 'POST'])
def get_comprehensive_feedback():
    """Get comprehensive feedback including performance analysis and improvement suggestions"""
    try:
        # Handle both GET and POST methods
        if request.method == 'GET':
            incident_id = request.args.get('incident_id')
            if not incident_id:
                return jsonify({'error': 'incident_id parameter required'}), 400
            
            # Find incident by ID
            incident_data = None
            for incident in fire_incidents:
                if str(incident.get('id')) == str(incident_id):
                    incident_data = incident
                    break
            
            if not incident_data:
                return jsonify({'error': f'Incident {incident_id} not found'}), 404
        else:
            # POST method
            data = request.json
            incident_data = data.get('incident_data', {})
       
        print(f"🔍 DEBUG - Comprehensive feedback requested for incident: {incident_data}")
       
        if not fire_incidents or len(fire_incidents) < 2:
            return jsonify({
                'status': 'no_data',
                'message': 'Not enough historical data for analysis. Need at least 2 incidents.',
                'performance_analysis': {
                    'status': 'no_data',
                    'message': 'Record more incidents to enable performance comparison',
                    'color': 'gray',
                    'improvement_opportunity': 0
                },
                'improvement_suggestions': get_initial_data_suggestions(),
                'training_recommendations': [],
                'success_factors': [],
                'comparison_metrics': {
                    'current_response_time': incident_data.get('response_time', incident_data.get('response_time_min', 0)),
                    'expected_response_time': '--',
                    'time_difference': '--',
                    'performance_ratio': '--',
                    'similar_incidents_count': 0
                }
            })
       
        # Get detailed comparison
        comparison = fire_analyzer.compare_incident(incident_data, fire_incidents) if fire_analyzer else {}
        print(f"🔍 DEBUG - Comparison result: {comparison}")
       
        # Generate comprehensive feedback
        feedback = generate_comprehensive_feedback(incident_data, comparison, fire_incidents)
        print(f"✅ DEBUG - Generated comprehensive feedback")
       
        return jsonify(feedback)
       
    except Exception as e:
        print(f"❌ DEBUG - Error in comprehensive feedback: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}',
            'performance_analysis': {
                'status': 'error',
                'message': 'Analysis temporarily unavailable',
                'color': 'gray',
                'improvement_opportunity': 0
            },
            'improvement_suggestions': [],
            'training_recommendations': [],
            'success_factors': [],
            'comparison_metrics': {}
        }), 500
@app.route('/api/debug-features', methods=['POST'])
def debug_features():
    """Debug endpoint to check feature consistency"""
    try:
        data = request.json
        incident_data = data.get('incident_data', {})
     
        if not fire_analyzer:
            return jsonify({'error': 'Analyzer not ready'})
     
        # Prepare features exactly like in prediction
        features_df = pd.DataFrame([incident_data])
     
        required_fields = {
            'severity': 'moderate',
            'weather': incident_data.get('weather_condition', 'Sunny'),
            'temperature': incident_data.get('temperature_c', 25),
            'humidity': incident_data.get('humidity_pct', 70),
            'wind_speed': incident_data.get('wind_speed_kmh', 10),
            'distance': incident_data.get('distance', 3.0),
            'response_time': incident_data.get('response_time_min', 5),
            'timestamp': incident_data.get('timestamp', datetime.now().isoformat()),
            'type': incident_data.get('type_of_occupancy', 'Other')
        }
     
        for field, default_value in required_fields.items():
            if field not in features_df.columns:
                features_df[field] = default_value
     
        features_processed = fire_analyzer.preprocess_data(features_df, for_classification=False)
     
        return jsonify({
            'input_features': list(features_df.columns),
            'processed_features': list(features_processed.columns),
            'regression_features_expected': fire_analyzer.regression_features,
            'classification_features_expected': fire_analyzer.classification_features,
            'regression_features_available': [f for f in fire_analyzer.regression_features if f in features_processed.columns],
            'classification_features_available': [f for f in fire_analyzer.classification_features if f in features_processed.columns]
        })
     
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Simple test endpoint to check if backend is working"""
    return jsonify({
        'status': 'success',
        'message': 'Backend is working!',
        'timestamp': datetime.now().isoformat(),
        'incidents_count': len(fire_incidents)
    })
@app.route('/api/test-feedback', methods=['POST'])
def test_feedback():
    """Test endpoint for feedback system"""
    test_data = {
        'performance_analysis': {
            'status': 'good',
            'message': '✅ Good performance! Faster than average response.',
            'color': 'blue',
            'efficiency_rating': 'high',
            'efficiency_score': 1.5,
            'performance_ratio': 0.85,
            'improvement_opportunity': 0
        },
        'improvement_suggestions': [
            {
                'category': 'response_time',
                'priority': 'low',
                'title': '🚀 Excellent Response Time',
                'description': 'Your response was faster than similar historical incidents.',
                'actionable_steps': [
                    'Continue current response procedures',
                    'Document successful strategies',
                    'Share best practices with team'
                ],
                'expected_impact': 'Maintain excellent performance',
                'implementation_difficulty': 'low',
                'time_to_implement': 'Ongoing'
            }
        ],
        'training_recommendations': [
            {
                'type': 'data_collection',
                'priority': 'low',
                'title': '📈 Continue Data Collection',
                'description': 'Current data quality is good. Continue recording incidents.',
                'actions': [
                    'Maintain current data recording practices',
                    'Include varied incident types',
                    'Track all response metrics consistently'
                ],
                'benefits': 'Improve long-term analysis accuracy'
            }
        ],
        'success_factors': [
            {
                'factor': 'exceptional_speed',
                'message': '🚀 Response was 2.5 minutes faster than similar incidents!',
                'best_practices': [
                    'Document the strategies used in this response',
                    'Share successful approaches with other teams',
                    'Consider making these practices standard'
                ]
            }
        ],
        'comparison_metrics': {
            'current_response_time': 5,
            'expected_response_time': 7.5,
            'time_difference': -2.5,
            'performance_ratio': 0.67,
            'similar_incidents_count': 8
        }
    }
    return jsonify(test_data)

# =============================================
# HYDRANTS MANAGEMENT ENDPOINTS
# =============================================
@app.route('/api/hydrants', methods=['GET', 'POST'])
def manage_hydrants():
    """Get all hydrants or add new hydrant"""
    global fire_hydrants
    
    if request.method == 'GET':
        return jsonify({'hydrants': fire_hydrants})
    
    elif request.method == 'POST':
        data = request.json
        hydrant = {
            'id': len(fire_hydrants) + 1,
            'number': data.get('number'),
            'address': data.get('address'),
            'latitude': float(data.get('latitude')),
            'longitude': float(data.get('longitude')),
            'status': data.get('status', 'operational'),
            'remarks': data.get('remarks', ''),
            'created_at': datetime.now().isoformat()
        }
        fire_hydrants.append(hydrant)
        save_hydrants_to_file()
        return jsonify({'success': True, 'hydrant': hydrant})

@app.route('/api/hydrants/<int:hydrant_id>', methods=['PUT', 'DELETE'])
def update_delete_hydrant(hydrant_id):
    """Update or delete a specific hydrant"""
    global fire_hydrants
    
    if request.method == 'PUT':
        data = request.json
        for hydrant in fire_hydrants:
            if hydrant['id'] == hydrant_id:
                hydrant.update({
                    'number': data.get('number', hydrant['number']),
                    'address': data.get('address', hydrant['address']),
                    'latitude': float(data.get('latitude', hydrant['latitude'])),
                    'longitude': float(data.get('longitude', hydrant['longitude'])),
                    'status': data.get('status', hydrant['status']),
                    'remarks': data.get('remarks', hydrant['remarks']),
                    'updated_at': datetime.now().isoformat()
                })
                save_hydrants_to_file()
                return jsonify({'success': True, 'hydrant': hydrant})
        return jsonify({'success': False, 'error': 'Hydrant not found'}), 404
    
    elif request.method == 'DELETE':
        fire_hydrants = [h for h in fire_hydrants if h['id'] != hydrant_id]
        save_hydrants_to_file()
        return jsonify({'success': True})

def save_hydrants_to_file():
    """Save hydrants to JSON file"""
    try:
        hydrants_file = os.path.join(SCRIPT_DIR, 'fire-hydrants.json')
        with open(hydrants_file, 'w') as f:
            json.dump(fire_hydrants, f, indent=2)
    except Exception as e:
        print(f"Error saving hydrants: {e}")

def load_hydrants_from_file():
    """Load hydrants from JSON file"""
    global fire_hydrants
    try:
        hydrants_file = os.path.join(SCRIPT_DIR, 'fire-hydrants.json')
        if os.path.exists(hydrants_file):
            with open(hydrants_file, 'r') as f:
                fire_hydrants = json.load(f)
            print(f"Loaded {len(fire_hydrants)} hydrants from file")
        else:
            print(f"Hydrants file not found at: {hydrants_file}")
            fire_hydrants = []
    except Exception as e:
        print(f"Error loading hydrants: {e}")
        fire_hydrants = []

# =============================================
# HAZARD ROADS MANAGEMENT ENDPOINTS
# =============================================
@app.route('/api/hazard-roads', methods=['GET', 'POST'])
def manage_hazard_roads():
    """Get all hazard roads or add new hazard road"""
    global hazard_roads
    
    if request.method == 'GET':
        return jsonify({'hazard_roads': hazard_roads})
    
    elif request.method == 'POST':
        data = request.json
        hazard = {
            'id': len(hazard_roads) + 1,
            'name': data.get('name'),
            'coordinates': data.get('coordinates'),
            'reason': data.get('reason', 'Road too narrow for fire trucks'),
            'severity': data.get('severity', 'high'),
            'created_at': datetime.now().isoformat()
        }
        hazard_roads.append(hazard)
        save_hazard_roads_to_file()
        return jsonify({'success': True, 'hazard_road': hazard})

@app.route('/api/hazard-roads/<int:hazard_id>', methods=['DELETE'])
def delete_hazard_road(hazard_id):
    """Delete a specific hazard road"""
    global hazard_roads
    hazard_roads = [h for h in hazard_roads if h['id'] != hazard_id]
    save_hazard_roads_to_file()
    return jsonify({'success': True})

def save_hazard_roads_to_file():
    """Save hazard roads to JSON file"""
    try:
        with open('hazard-roads.json', 'w') as f:
            json.dump(hazard_roads, f, indent=2)
    except Exception as e:
        print(f"Error saving hazard roads: {e}")

def load_hazard_roads_from_file():
    """Load hazard roads from JSON file"""
    global hazard_roads
    try:
        if os.path.exists('hazard-roads.json'):
            with open('hazard-roads.json', 'r') as f:
                hazard_roads = json.load(f)
            print(f"Loaded {len(hazard_roads)} hazard roads from file")
        else:
            hazard_roads = [
                {
                    'id': 1,
                    'name': 'Narrow Alley near Public Market',
                    'coordinates': [[14.2791, 121.4160], [14.2795, 121.4165]],
                    'reason': 'Road too narrow for fire trucks (2m width)',
                    'severity': 'high'
                },
                {
                    'id': 2,
                    'name': 'Flooded Road during Rainy Season - Barangay 1',
                    'coordinates': [[14.2800, 121.4170], [14.2810, 121.4180]],
                    'reason': 'Frequently flooded, impassable during heavy rain',
                    'severity': 'medium'
                }
            ]
            save_hazard_roads_to_file()
    except Exception as e:
        print(f"Error loading hazard roads: {e}")
        hazard_roads = []

if __name__ == '__main__':
    print("Starting Enhanced Fire Incident Analysis Backend...")
    print("API Endpoints:")
    print(" GET /api/health - Health check")
    print(" GET /api/model-status - Model status with accuracy")
    print(" POST /api/compare - Compare incident")
    print(" GET /api/performance-trends - Performance trends")
    print(" POST /api/train - Train analyzer")
    print(" GET /api/incidents - Get incidents")
    print(" POST /api/incidents - Store incident")
    print(" GET /api/statistics - Statistics")
    print(" POST /api/load-csv - Load CSV")
    print(" GET /api/analyzer-status - Analyzer status")
    print(" POST /api/predict-category - Predict response category")
    print(" POST /api/improvement-suggestions - Get improvement suggestions")
    print(" POST /api/performance-feedback - Get performance feedback")
    print(" GET /api/training-feedback - Get training feedback")
    print(" POST /api/comprehensive-feedback - Get comprehensive feedback")
    print(" POST /api/debug-features - Debug feature consistency")
    print(" GET /api/test - Test backend connectivity")
    print(" POST /api/test-feedback - Test feedback system")
    print(" GET /api/hydrants - Get hydrants")
    print(" POST /api/hydrants - Add hydrant")
    print(" PUT /api/hydrants/<id> - Update hydrant")
    print(" DELETE /api/hydrants/<id> - Delete hydrant")
    print(" GET /api/hazard-roads - Get hazard roads")
    print(" POST /api/hazard-roads - Add hazard road")
    print(" DELETE /api/hazard-roads/<id> - Delete hazard road")
    print("\nServer running on http://localhost:5000")
    
    # Load hydrants and hazard roads
    load_hydrants_from_file()
    load_hazard_roads_from_file()
    
    app.run(debug=True, host='0.0.0.0', port=5000)