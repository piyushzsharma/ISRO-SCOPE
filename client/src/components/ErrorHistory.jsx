import { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const ErrorHistory = ({ satellite }) => {
  const [errorData, setErrorData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  console.log('ErrorHistory mounted with satellite:', satellite);

  useEffect(() => {
    console.log('useEffect running with satellite:', satellite?.id);
    
    const fetchErrorData = async () => {
      console.log('fetchErrorData called with satellite:', satellite?.id);
      
      if (!satellite) {
        console.log('No satellite selected, clearing error data');
        setErrorData([]);
        return;
      }

      setLoading(true);
      try {
        const response = await fetch(`http://localhost:5000/api/error-history?t=${Date.now()}`);
        if (!response.ok) {
          throw new Error('Failed to fetch error history');
        }
        
        const data = await response.json();
        console.log('Error history data:', data); // Debug log
        
        if (!data.success) {
          throw new Error(data.message || 'Failed to load error data');
        }

        // Map UI satellite IDs to data keys
        const satelliteIdMap = {
          'meo-01': 'meo1',
          'meo-02': 'meo2',
          'geo-01': 'geo',
          'gso-01': 'gso'
        };
        
        // Try to find the correct data key for the current satellite
        const satelliteKey = satelliteIdMap[satellite.id.toLowerCase()] || satellite.id.toLowerCase();
        
        console.log('Satellite ID:', satellite.id);
        console.log('Looking for data with key:', satelliteKey);
        console.log('Available keys in response:', Object.keys(data.data));
        
        const satelliteErrors = data.data[satelliteKey] || [];
        console.log(`Found ${satelliteErrors.length} error records for ${satelliteKey}`);
        
        if (satelliteErrors.length > 0) {
          console.log('First error record:', satelliteErrors[0]);
        } else {
          console.warn('No error data found for satellite:', satellite.id, 'using key:', satelliteKey);
        }
        
        // Process and format the data for the chart
        const formattedData = satelliteErrors.map(error => {
          try {
            const utcTime = error.utc_time || error.timestamp;
            const date = new Date(utcTime);
            
            // If the date is invalid, log the problematic date string
            if (isNaN(date.getTime())) {
              console.warn('Invalid date string:', utcTime);
            }
            
            return {
              timestamp: date,
              date: date.toLocaleDateString('en-US', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
              }),
              positionError: Math.abs(parseFloat(error.position_error || 0)),
              clockError: Math.abs(parseFloat(error['satclockerror (m)'] || error.clockError || 0)),
              xError: Math.abs(parseFloat(error['x_error (m)'] || error.xError || 0)),
              yError: Math.abs(parseFloat(error['y_error (m)'] || error.yError || 0)),
              zError: Math.abs(parseFloat(error['z_error (m)'] || error.zError || 0))
            };
          } catch (e) {
            console.error('Error processing error record:', error, e);
            return null;
          }
        }).filter(Boolean); // Remove any null entries from errors

        // Sort by timestamp
        formattedData.sort((a, b) => a.timestamp - b.timestamp);
        console.log('Formatted data:', formattedData);

        setErrorData(formattedData);
        setError(formattedData.length === 0 ? 'No error data available for the selected satellite' : null);
      } catch (err) {
        console.error('Error fetching error history:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchErrorData();
  }, [satellite?.id]);

  // Custom tooltip component
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-gray-800 p-3 border border-gray-700 rounded shadow-lg">
          <p className="font-semibold text-white">{data.date}</p>
          <div className="grid grid-cols-2 gap-2 mt-2">
            <span className="text-gray-300">Position Error:</span>
            <span className="text-right">{data.positionError.toFixed(3)} m</span>
            
            <span className="text-gray-300">Clock Error:</span>
            <span className="text-right">{data.clockError.toFixed(3)} m</span>
            
            <span className="text-gray-300">X Error:</span>
            <span className="text-right">{data.xError.toFixed(3)} m</span>
            
            <span className="text-gray-300">Y Error:</span>
            <span className="text-right">{data.yError.toFixed(3)} m</span>
            
            <span className="text-gray-300">Z Error:</span>
            <span className="text-right">{data.zError.toFixed(3)} m</span>
          </div>
        </div>
      );
    }
    return null;
  };

  // If there's an error or no data, show the appropriate message
  if (error) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-4 text-center">
        <div className="bg-red-900 bg-opacity-50 rounded-full p-3 mb-3">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-6 w-6 text-red-400" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" 
            />
          </svg>
        </div>
        <p className="text-red-400">Error loading data</p>
        <p className="text-sm text-red-500 mt-1">{error}</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!satellite) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-4">
        <div className="bg-blue-900 bg-opacity-50 rounded-full p-4 mb-3">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-8 w-8 text-blue-400" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M12 6v2m0 10h.01M12 18h.01M12 10a1 1 0 100-2 1 1 0 000 2z" 
            />
          </svg>
        </div>
        <p className="text-gray-400">Select a satellite from the 3D view</p>
        <p className="text-sm text-gray-500 mt-2">
          Click on any satellite marker to view its error history
        </p>
      </div>
    );
  }

  if (errorData.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-4 text-center">
        <div className="bg-yellow-900 bg-opacity-50 rounded-full p-3 mb-3">
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            className="h-6 w-6 text-yellow-400" 
            fill="none" 
            viewBox="0 0 24 24" 
            stroke="currentColor"
          >
            <path 
              strokeLinecap="round" 
              strokeLinejoin="round" 
              strokeWidth={2} 
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" 
            />
          </svg>
        </div>
        <p className="text-yellow-400">No error data available</p>
        <p className="text-sm text-yellow-500 mt-1">
          No errors recorded for {satellite.id} in the selected time range
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-lg font-semibold mb-3 text-white">
        Error History - {satellite.id}
      </h3>
      <div className="flex-1 overflow-x-auto">
        <div className="h-64 mb-4" style={{ minWidth: `${Math.max(800, errorData.length * 10)}px` }}>
          <h4 className="text-sm font-medium text-gray-300 mb-2">Position Error (m)</h4>
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={errorData}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis 
                  dataKey="date" 
                  stroke="#a0aec0"
                  tick={{ fontSize: 10 }}
                  interval={Math.ceil(errorData.length / 10)} // Show fewer ticks
                  minTickGap={20} // Add more gap between ticks
                />
                <YAxis 
                  stroke="#a0aec0"
                  tick={{ fontSize: 10 }}
                  width={40}
                  domain={[0, 'dataMax']}
                  tickFormatter={(value) => value.toFixed(2)}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="positionError"
                  stroke="#ff4757" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#ff6b81' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="h-32" style={{ minWidth: `${Math.max(800, errorData.length * 10)}px` }}>
          <h4 className="text-sm font-medium text-gray-300 mb-2">Clock Error (m)</h4>
          <div className="h-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={errorData}
                margin={{ top: 5, right: 5, left: 5, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                <XAxis 
                  dataKey="date" 
                  stroke="#a0aec0"
                  tick={{ fontSize: 10 }}
                  interval={Math.ceil(errorData.length / 10)} // Show fewer ticks
                  minTickGap={20} // Add more gap between ticks
                />
                <YAxis 
                  stroke="#a0aec0"
                  tick={{ fontSize: 10 }}
                  width={40}
                  domain={[0, 'dataMax']}
                  tickFormatter={(value) => value.toFixed(3)}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line 
                  type="monotone" 
                  dataKey="clockError"
                  stroke="#4dabf7" 
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 3, fill: '#74c0fc' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      <div className="mt-2 text-xs text-gray-400 flex justify-between">
        <span>Showing {errorData.length} data points for {satellite.id}</span>
        <div className="flex items-center">
          <span className="flex items-center mr-4">
            <span className="w-3 h-3 bg-red-500 rounded-full mr-1"></span>
            <span>Position Error</span>
          </span>
          <span className="flex items-center">
            <span className="w-3 h-3 bg-blue-400 rounded-full mr-1"></span>
            <span>Clock Error</span>
          </span>
        </div>
      </div>
    </div>
  );
};

export default ErrorHistory;
