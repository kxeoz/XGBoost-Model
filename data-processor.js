// Fire incident data processor
class FireDataProcessor {
    constructor() {
        this.incidents = [];
        this.processedData = [];
        this.barangayStats = {};
    }

    // Load and parse CSV data
    async loadIncidentData() {
        try {
            // In a real implementation, you would fetch the CSV file
            // For now, we'll create a sample from the provided data structure
            this.incidents = this.parseCSVData();
            this.processHistoricalData();
            console.log(`Loaded ${this.incidents.length} fire incidents`);
            return this.incidents;
        } catch (error) {
            console.error('Error loading incident data:', error);
            return [];
        }
    }

    // Parse the CSV structure into usable data
    parseCSVData() {
        // This would normally parse actual CSV data
        // For now, we'll create structured data based on the CSV format
        return [
            {
                station: "Santa Cruz, Laguna",
                date: "1/20/2010",
                location: "Brgy. Gatid, Santa Cruz, Laguna",
                responseTime: 5.0,
                distance: 4.6,
                occupancy: "Business",
                weather: "Rainy",
                temperature: 27.4,
                humidity: 84,
                windSpeed: 14.2,
                barangay: "Gatid"
            },
            {
                station: "Santa Cruz, Laguna",
                date: "2/8/2010",
                location: "Brgy. Villa Silangan, Santa Cruz Laguna",
                responseTime: 5.0,
                distance: 3.5,
                occupancy: "Grass",
                weather: "Sunny",
                temperature: 26.3,
                humidity: 80,
                windSpeed: 13.9,
                barangay: "Villa Silangan"
            },
            // Add more sample data based on your CSV structure
            // In production, this would parse the actual CSV file
        ];
    }

    // Process historical data for statistics
    processHistoricalData() {
        this.barangayStats = {};
        
        this.incidents.forEach(incident => {
            const barangay = this.extractBarangay(incident.location);
            
            if (!this.barangayStats[barangay]) {
                this.barangayStats[barangay] = {
                    totalIncidents: 0,
                    avgResponseTime: 0,
                    commonOccupancies: {},
                    weatherPatterns: {},
                    totalResponseTime: 0
                };
            }
            
            const stats = this.barangayStats[barangay];
            stats.totalIncidents++;
            stats.totalResponseTime += incident.responseTime;
            stats.avgResponseTime = stats.totalResponseTime / stats.totalIncidents;
            
            // Count occupancy types
            stats.commonOccupancies[incident.occupancy] = 
                (stats.commonOccupancies[incident.occupancy] || 0) + 1;
                
            // Count weather patterns
            stats.weatherPatterns[incident.weather] = 
                (stats.weatherPatterns[incident.weather] || 0) + 1;
        });
    }

    // Extract barangay from location string
    extractBarangay(location) {
        if (!location) return 'Unknown';
        
        const brgyMatch = location.match(/Brgy\.?\s*([^,]+)/i);
        if (brgyMatch) {
            return brgyMatch[1].trim();
        }
        
        // Try other patterns
        const patterns = [
            /Brgy\.?\s*(\w+)/i,
            /Barangay\s*(\w+)/i,
            /B\.?\s*(\w+)/i
        ];
        
        for (const pattern of patterns) {
            const match = location.match(pattern);
            if (match) return match[1].trim();
        }
        
        return 'Unknown';
    }

    // Get incidents near a location
    getNearbyIncidents(lat, lng, radiusKm = 2) {
        // For demo purposes, return incidents based on barangay proximity
        // In real implementation, use actual coordinates
        const simulatedBarangays = ['Gatid', 'Villa Silangan', 'Calios', 'Duhat', 'Bagumbayan'];
        const randomBarangay = simulatedBarangays[Math.floor(Math.random() * simulatedBarangays.length)];
        
        return this.incidents.filter(incident => 
            this.extractBarangay(incident.location) === randomBarangay
        ).slice(0, 5); // Return max 5 incidents
    }

    // Get statistics for a specific area
    getAreaStats(barangay) {
        return this.barangayStats[barangay] || {
            totalIncidents: 0,
            avgResponseTime: 0,
            commonOccupancies: {},
            weatherPatterns: {}
        };
    }

    // Get risk factors for prediction
    getRiskFactors(location, weatherData, occupancyType) {
        const barangay = this.extractBarangay(location);
        const stats = this.getAreaStats(barangay);
        
        return {
            historicalIncidents: Math.min(stats.totalIncidents / 10, 1), // Normalize to 0-1
            weatherRisk: this.calculateWeatherRisk(weatherData),
            occupancyRisk: this.calculateOccupancyRisk(occupancyType),
            responseTime: stats.avgResponseTime || 5.0,
            barangay: barangay
        };
    }

    calculateWeatherRisk(weatherData) {
        if (!weatherData) return 0.5;
        
        let risk = 0.5; // Base risk
        
        // Temperature factor (higher temp = higher risk)
        if (weatherData.temperature > 35) risk += 0.3;
        else if (weatherData.temperature > 30) risk += 0.2;
        else if (weatherData.temperature > 25) risk += 0.1;
        
        // Humidity factor (lower humidity = higher risk)
        if (weatherData.humidity < 30) risk += 0.2;
        else if (weatherData.humidity < 50) risk += 0.1;
        
        // Weather condition
        if (weatherData.condition.includes('Storm')) risk += 0.3;
        else if (weatherData.condition.includes('Rain')) risk -= 0.2;
        
        return Math.max(0, Math.min(1, risk));
    }

    calculateOccupancyRisk(occupancyType) {
        const riskLevels = {
            'Residential': 0.3,
            'Business': 0.6,
            'Industrial': 0.8,
            'Educational': 0.4,
            'Mercantile': 0.7,
            'Storage': 0.9,
            'Grass': 0.2,
            'Other': 0.5
        };
        
        return riskLevels[occupancyType] || 0.5;
    }
}

// Create global instance
const fireDataProcessor = new FireDataProcessor();