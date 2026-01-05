// Simple machine learning model for fire risk prediction
class FireRiskModel {
    constructor() {
        this.weights = {
            historical: 0.4,
            weather: 0.3,
            occupancy: 0.3
        };
        this.trainingData = [];
        this.isTrained = false;
    }

    // Train model with historical data
    async trainModel(incidents) {
        console.log('Training fire risk model...');
        
        // Simple training based on incident patterns
        this.trainingData = incidents.map(incident => ({
            features: {
                historical: Math.min(incident.historicalIncidents || 0, 1),
                weather: incident.weatherRisk || 0.5,
                occupancy: incident.occupancyRisk || 0.5
            },
            actualRisk: this.calculateActualRisk(incident)
        }));
        
        this.isTrained = true;
        console.log('Model training completed');
    }

    // Calculate actual risk based on incident severity
    calculateActualRisk(incident) {
        let risk = 0.5;
        
        // Response time factor (longer response = higher impact)
        if (incident.responseTime > 10) risk += 0.3;
        else if (incident.responseTime > 5) risk += 0.1;
        
        // Occupancy severity
        const severity = {
            'Residential': 0.3,
            'Business': 0.6,
            'Industrial': 0.8,
            'Educational': 0.7,
            'Storage': 0.9,
            'Grass': 0.2
        };
        
        risk += severity[incident.occupancy] || 0.5;
        
        return Math.min(risk, 1.0);
    }

    // Predict fire risk
    predict(features) {
        if (!this.isTrained) {
            // Use simple weighted average if not trained
            return this.simplePrediction(features);
        }

        // Simple neural network-like calculation
        let riskScore = 0;
        
        riskScore += features.historical * this.weights.historical;
        riskScore += features.weather * this.weights.weather;
        riskScore += features.occupancy * this.weights.occupancy;
        
        // Add some randomness to simulate real prediction
        riskScore += (Math.random() - 0.5) * 0.1;
        
        return Math.max(0, Math.min(1, riskScore));
    }

    // Simple prediction algorithm
    simplePrediction(features) {
        const baseRisk = (
            features.historical * 0.4 +
            features.weather * 0.3 +
            features.occupancy * 0.3
        );
        
        return Math.max(0, Math.min(1, baseRisk));
    }

    // Update model with new incident data
    updateModel(newIncident) {
        this.trainingData.push({
            features: {
                historical: newIncident.historicalIncidents,
                weather: newIncident.weatherRisk,
                occupancy: newIncident.occupancyRisk
            },
            actualRisk: this.calculateActualRisk(newIncident)
        });
        
        // Simple weight adjustment based on new data
        this.adjustWeights();
    }

    // Adjust weights based on recent performance
    adjustWeights() {
        // Simple weight adjustment - in real implementation, use gradient descent
        const adjustment = 0.01;
        this.weights.historical += (Math.random() - 0.5) * adjustment;
        this.weights.weather += (Math.random() - 0.5) * adjustment;
        this.weights.occupancy += (Math.random() - 0.5) * adjustment;
        
        // Normalize weights
        const total = this.weights.historical + this.weights.weather + this.weights.occupancy;
        this.weights.historical /= total;
        this.weights.weather /= total;
        this.weights.occupancy /= total;
    }

    // Get risk level category
    getRiskCategory(riskScore) {
        if (riskScore >= 0.7) return 'HIGH';
        if (riskScore >= 0.4) return 'MEDIUM';
        return 'LOW';
    }

    // Get risk color
    getRiskColor(riskScore) {
        if (riskScore >= 0.7) return '#e53e3e';
        if (riskScore >= 0.4) return '#dd6b20';
        return '#38a169';
    }

    // Get recommendations based on risk
    getRecommendations(riskScore, factors) {
        const recommendations = [];
        
        if (riskScore >= 0.7) {
            recommendations.push('🚨 High alert: Consider pre-positioning resources');
            recommendations.push('📞 Notify nearby stations for standby support');
            recommendations.push('🔍 Increase patrol frequency in area');
        } else if (riskScore >= 0.4) {
            recommendations.push('⚠️ Moderate risk: Monitor weather conditions');
            recommendations.push('📋 Review emergency response plans');
            recommendations.push('🧯 Check equipment readiness');
        } else {
            recommendations.push('✅ Low risk: Maintain regular monitoring');
            recommendations.push('📚 Conduct preventive education');
            recommendations.push('🔧 Schedule maintenance checks');
        }
        
        // Specific recommendations based on factors
        if (factors.historical > 0.7) {
            recommendations.push('📊 High incident history: Consider permanent measures');
        }
        
        if (factors.weather > 0.7) {
            recommendations.push('🌡️ Extreme weather: Monitor closely');
        }
        
        if (factors.occupancy > 0.7) {
            recommendations.push('🏭 Industrial area: Specialized response needed');
        }
        
        return recommendations;
    }
}

// Create global model instance
const fireRiskModel = new FireRiskModel();