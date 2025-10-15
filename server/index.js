const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const XLSX = require('xlsx');

// Function to read error history from CSV file
function readErrorHistory(satelliteType) {
  try {
    const filePath = path.join(__dirname, 'data', 'error_history', `${satelliteType}_errors.csv`);
    
    if (!fs.existsSync(filePath)) {
      console.warn(`No error history found for ${satelliteType}`);
      return [];
    }
    
    // Read the CSV file
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    const lines = fileContent.split('\n').filter(line => line.trim() !== '');
    const headers = lines[0].split(',').map(h => h.trim());
    
    // Parse CSV rows
    const data = [];
    for (let i = 1; i < lines.length; i++) {
      if (!lines[i].trim()) continue; // Skip empty lines
      
      // Handle CSV values that might contain commas (not in the current data, but good practice)
      const values = lines[i].match(/\s*([^,]+|'[^']*'|"[^"]*")\s*(?:,|$)/g)
        .map(v => v.replace(/\s*,\s*$/, '').trim().replace(/^["']|["']$/g, ''));
      
      const entry = {};
      
      headers.forEach((header, index) => {
        entry[header] = values[index] || '';
      });
      
      // Convert timestamp to Date object and parse numeric values
      if (entry.utc_time) {
        // Parse the date in format M/D/YYYY H:MM
        const [datePart, timePart] = entry.utc_time.split(' ');
        const [month, day, year] = datePart.split('/').map(Number);
        const [hours, minutes] = timePart.split(':').map(Number);
        
        // Create date in UTC
        entry.timestamp = new Date(Date.UTC(year, month - 1, day, hours, minutes));
      }
      
      // Parse numeric values
      const numericFields = ['x_error (m)', 'y_error (m)', 'z_error (m)', 'satclockerror (m)'];
      numericFields.forEach(field => {
        if (entry[field] !== undefined) {
          entry[field] = parseFloat(entry[field]) || 0;
        }
      });
      
      // Calculate total position error (Euclidean distance)
      if (entry['x_error (m)'] !== undefined && 
          entry['y_error (m)'] !== undefined && 
          entry['z_error (m)'] !== undefined) {
        entry.position_error = Math.sqrt(
          Math.pow(entry['x_error (m)'], 2) + 
          Math.pow(entry['y_error (m)'], 2) + 
          Math.pow(entry['z_error (m)'], 2)
        ).toFixed(3);
      }
      
      // Add satellite type
      entry.satellite = satelliteType.toUpperCase();
      
      data.push(entry);
    }
    
    return data;
  } catch (error) {
    console.error(`Error reading error history for ${satelliteType}:`, error);
    return [];
  }
}

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors());
app.use(express.json());

// Prevent caching of API responses
app.use((req, res, next) => {
  res.header('Cache-Control', 'no-cache, no-store, must-revalidate');
  res.header('Pragma', 'no-cache');
  res.header('Expires', '0');
  next();
});

// Mock dashboard data
const dashboardData = {
  "kpi_data": {
    "accuracy": 0.15,
    "high_error_count": 3,
    "model_status": "ONLINE & PREDICTING",
    "last_update": "2024-03-08 00:00:00 UTC"
  },
  "satellite_positions": [
    {
      "id": "MEO-01",
      "type": "MEO",
      "position": [20.5937, 78.9629],
      "status_color": "green",
      "error_data": { "ephemeris_error": "0.45m", "clock_error": "0.8ns" }
    },
    {
      "id": "MEO-02",
      "type": "MEO",
      "position": [34.0522, -118.2437],
      "status_color": "red",
      "error_data": { "ephemeris_error": "2.92m", "clock_error": "4.1ns" }
    },
    {
      "id": "GEO-01",
      "type": "GEO",
      "position": [0, -75],
      "status_color": "green",
      "error_data": { "ephemeris_error": "0.12m", "clock_error": "0.4ns" }
    }
  ],
  "charts_data": {
    "constellation_distribution": {
      "labels": ["MEO", "MEO", "MEO", "280th", "GEO/GSO"],
      "values": [25, 60, 45, 85, 75]
    },
    "error_histogram": {
      "labels": ["001th", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "Normality Score", "", "", "", "", "", "", "", "", "9001th"],
      "values": [0, 5, 10, 18, 28, 40, 55, 70, 85, 100, 120, 140, 160, 180, 189, 195, 198, 199, 200, 199, 198, 195, 189, 180, 160, 140, 120, 100, 85, 70, 55, 40, 28, 18, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    },
    "error_trend": {
      "labels": ["-40th", "-20th", "0th", "20th", "40th", "60th", "80th", "100th", "120th", "140th", "160th", "180th", "200th", "220th", "240th"],
      "values": [2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 8.5, 9, 9.2, 9.5, 9.8, 10]
    }
  }
};

// Function to get the last modification time of the most recently changed file
function getLastCodeUpdate() {
  const directories = [
    path.join(__dirname, ''),
    path.join(__dirname, '..', 'client', 'src')
  ];

  let latestTime = 0;

  directories.forEach(dir => {
    try {
      const files = fs.readdirSync(dir, { recursive: true });
      files.forEach(file => {
        const filePath = path.join(dir, file);
        if (fs.statSync(filePath).isFile()) {
          const stats = fs.statSync(filePath);
          if (stats.mtime > latestTime) {
            latestTime = stats.mtime;
          }
        }
      });
    } catch (error) {
      console.log('Error reading directory:', dir, error.message);
    }
  });

  return latestTime;
}

// Function to format the timestamp
function formatLastUpdate(timestamp) {
  if (timestamp === 0) return 'Never';

  const now = new Date();
  const diffMs = now - timestamp;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffMins < 1) {
    return 'Just now';
  } else if (diffMins < 60) {
    return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
  } else {
    return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
  }
}

// API endpoint
// Get error history for all satellites
app.get('/api/error-history', (req, res) => {
  console.log('Received request for error history');
  try {
    // Define all satellite types we expect to handle
    const satelliteTypes = ['meo1', 'meo2', 'geo', 'gso'];
    const responseData = {
      success: true,
      data: {}
    };

    // Load data for each satellite type
    satelliteTypes.forEach(satType => {
      const errors = readErrorHistory(satType);
      console.log(`Loaded ${errors.length} ${satType.toUpperCase()} error records`);
      
      // Only include in response if we have data
      if (errors.length > 0) {
        responseData.data[satType] = errors;
        console.log(`Sample ${satType} error record:`, JSON.stringify(errors[0], null, 2));
      } else {
        console.warn(`No data found for satellite type: ${satType}`);
      }
    });

    // If no data was loaded for any satellite, return an error
    if (Object.keys(responseData.data).length === 0) {
      throw new Error('No error history data found for any satellite');
    }

    res.json(responseData);
  } catch (error) {
    console.error('Error in /api/error-history:', error);
    res.status(500).json({
      success: false,
      message: 'Failed to load error history',
      error: error.message,
      stack: process.env.NODE_ENV === 'development' ? error.stack : undefined
    });
  }
});

// API endpoint
app.get('/api/dashboard-data', (req, res) => {
  const lastUpdateTimestamp = getLastCodeUpdate();
  dashboardData.kpi_data.last_update = formatLastUpdate(lastUpdateTimestamp);
  
  // Calculate high error count dynamically
  dashboardData.kpi_data.high_error_count = dashboardData.satellite_positions
    .filter(sat => sat.status_color === 'red').length;
    
  res.json(dashboardData);
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 G-ForcEST Backend Server running on port ${PORT}`);
  console.log(`📊 API available at http://localhost:${PORT}/api/dashboard-data`);
});
