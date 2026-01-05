
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from  import FireIncidentAnalyzer, load_incidents_from_csv, fire_incidents

# Instantiate analyzer
analyzer = FireIncidentAnalyzer()

# Load data
load_incidents_from_csv()

from  import fire_incidents # Re-import to get populated data
if not fire_incidents:
    print("No incidents loaded. Checking if we need to call load_incidents_from_csv explicitly.")
    # In app.py, load_incidents_from_csv() updates the global fire_incidents
    # But since we imported fire_incidents before calling it (in the line above), it might be empty.
    # Actually, from app import fire_incidents imports the object. If it's a list, it's mutable.
    # But let's check.
    
    # Let's inspect app.py's load_incidents_from_csv
    # It does: global fire_incidents; fire_incidents = [] ...
    # This rebinds the name, so the imported reference won't see the change if we imported it before.
    # We should inspect the module directly.
    import 
    .load_incidents_from_csv()
    incidents = .fire_incidents
else:
    incidents = fire_incidents

print(f"Loaded {len(incidents)} incidents.")

if len(incidents) < 5:
    print("Not enough incidents to train.")
    sys.exit(1)

# Train
print("Starting training...")
performance = analyzer.train_analyzer(incidents)
print("Training complete.")
print("Performance:", performance)

# Verify pickle file exists
if os.path.exists('fire_incident_analyzer.pkl'):
    print("fire_incident_analyzer.pkl saved successfully.")
else:
    print("Error: fire_incident_analyzer.pkl not found.")
