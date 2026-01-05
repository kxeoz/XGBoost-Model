// Main fire prediction system
class FirePredictor {
    constructor() {
        this.currentPrediction = null;
        this.isInitialized = false;
    }

    // Initialize the prediction system
    async initialize() {
        try {
            // Load historical data
            await fireDataProcessor.loadIncidentData();
            
            // Train ML model
            await fireRiskModel.trainModel(fireDataProcessor.incidents);
            
            this.isInitialized = true;
            console.log('Fire prediction system initialized');
        } catch (error) {
            console.error('Error initializing fire prediction:', error);
        }
    }

    // Predict fire risk for a location
    async predictFireRisk(location, weatherData, occupancyType) {
        if (!this.isInitialized) {
            await this.initialize();
        }

        // Show loading state
        this.showLoading();

        // Simulate API call delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        try {
            // Get risk factors from historical data
            const riskFactors = fireDataProcessor.getRiskFactors(location, weatherData, occupancyType);
            
            // Predict risk using ML model
            const riskScore = fireRiskModel.predict(riskFactors);
            const riskCategory = fireRiskModel.getRiskCategory(riskScore);
            const riskColor = fireRiskModel.getRiskColor(riskScore);
            
            // Get recommendations
            const recommendations = fireRiskModel.getRecommendations(riskScore, riskFactors);
            
            // Get nearby historical incidents
            const nearbyIncidents = fireDataProcessor.getNearbyIncidents();
            
            this.currentPrediction = {
                riskScore,
                riskCategory,
                riskColor,
                riskFactors,
                recommendations,
                nearbyIncidents,
                timestamp: new Date()
            };

            // Display results
            this.displayPrediction();
            this.displayHistoricalIncidents();
            
            return this.currentPrediction;
        } catch (error) {
            console.error('Error predicting fire risk:', error);
            this.displayError();
        }
    }

    // Show loading state
    showLoading() {
        document.getElementById('prediction-loading').classList.remove('hidden');
        document.getElementById('prediction-result').classList.add('hidden');
    }

    // Display prediction results
    displayPrediction() {
        const prediction = this.currentPrediction;
        if (!prediction) return;

        document.getElementById('prediction-loading').classList.add('hidden');
        document.getElementById('prediction-result').classList.remove('hidden');

        const resultDiv = document.getElementById('prediction-result');
        
        // Update risk factors display
        this.updateRiskFactors(prediction.riskFactors);
        
        // Create prediction display
        resultDiv.innerHTML = `
            <div class="text-center p-4 rounded-lg ${this.getRiskClass(prediction.riskCategory)}">
                <div class="text-3xl font-bold mb-2" style="color: ${prediction.riskColor}">
                    ${Math.round(prediction.riskScore * 100)}%
                </div>
                <div class="prediction-badge mb-3" style="background: ${prediction.riskColor}; color: white;">
                    ${prediction.riskCategory} RISK
                </div>
                <p class="text-sm text-gray-700 mb-3">
                    ${this.getRiskDescription(prediction.riskCategory)}
                </p>
            </div>
            
            <div class="mt-4">
                <h4 class="font-semibold text-sm mb-2">Recommendations:</h4>
                <ul class="text-xs space-y-1">
                    ${prediction.recommendations.map(rec => 
                        `<li class="flex items-start">
                            <span class="mr-2">•</span>
                            <span>${rec}</span>
                        </li>`
                    ).join('')}
                </ul>
            </div>
            
            <div class="prediction-features text-center mt-3">
                <p>Based on analysis of historical data and current conditions</p>
            </div>
        `;
    }

    // Update risk factors visualization
    updateRiskFactors(factors) {
        // Historical incidents factor
        const historyPercent = Math.round(factors.historicalIncidents * 100);
        document.getElementById('factor-history').textContent = `${historyPercent}%`;
        document.getElementById('factor-history-bar').style.width = `${historyPercent}%`;
        
        // Weather factor
        const weatherPercent = Math.round(factors.weatherRisk * 100);
        document.getElementById('factor-weather').textContent = `${weatherPercent}%`;
        document.getElementById('factor-weather-bar').style.width = `${weatherPercent}%`;
        
        // Occupancy factor
        const occupancyPercent = Math.round(factors.occupancyRisk * 100);
        document.getElementById('factor-occupancy').textContent = `${occupancyPercent}%`;
        document.getElementById('factor-occupancy-bar').style.width = `${occupancyPercent}%`;
    }

    // Display historical incidents
    displayHistoricalIncidents() {
        const incidents = this.currentPrediction?.nearbyIncidents || [];
        const container = document.getElementById('historical-incidents');
        
        if (incidents.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <p class="text-gray-600 text-sm">No historical incidents found in this area</p>
                </div>
            `;
            return;
        }
        
        container.innerHTML = incidents.map(incident => `
            <div class="bg-gray-50 p-3 rounded-lg border-l-4 border-orange-500">
                <div class="flex justify-between items-start mb-1">
                    <span class="font-medium text-sm">${incident.barangay}</span>
                    <span class="text-xs text-gray-500">${incident.date}</span>
                </div>
                <div class="text-xs text-gray-600 mb-1">
                    ${incident.occupancy} • ${incident.weather}
                </div>
                <div class="text-xs">
                    <span class="text-gray-500">Response: ${incident.responseTime}min</span>
                </div>
            </div>
        `).join('');
    }

    // Display error state
    displayError() {
        document.getElementById('prediction-loading').classList.add('hidden');
        document.getElementById('prediction-result').innerHTML = `
            <div class="text-center p-4 text-red-600">
                <i data-feather="alert-circle" class="w-8 h-8 mx-auto mb-2"></i>
                <p>Unable to calculate risk prediction</p>
                <p class="text-sm text-gray-600 mt-1">Please try again later</p>
            </div>
        `;
        feather.replace();
    }

    // Get CSS class for risk level
    getRiskClass(riskCategory) {
        switch (riskCategory) {
            case 'HIGH': return 'risk-high';
            case 'MEDIUM': return 'risk-medium';
            case 'LOW': return 'risk-low';
            default: return '';
        }
    }

    // Get description for risk level
    getRiskDescription(riskCategory) {
        switch (riskCategory) {
            case 'HIGH': return 'Elevated fire risk detected. Immediate attention recommended.';
            case 'MEDIUM': return 'Moderate fire risk. Standard precautions advised.';
            case 'LOW': return 'Low fire risk. Maintain regular safety protocols.';
            default: return 'Risk assessment completed.';
        }
    }

    // Add new incident to training data
    async addNewIncident(incidentData) {
        if (!this.isInitialized) await this.initialize();
        
        // Process the new incident
        const riskFactors = fireDataProcessor.getRiskFactors(
            incidentData.location,
            incidentData.weather,
            incidentData.occupancy
        );
        
        // Update ML model
        fireRiskModel.updateModel({
            ...incidentData,
            ...riskFactors
        });
        
        console.log('New incident added to training data');
    }
}

// Create global instance
const firePredictor = new FirePredictor();

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    firePredictor.initialize();
});