import React from 'react';

const PredictionPanel = ({ satellite }) => {
  // Error values for different satellite types and time intervals (in meters)
  const satelliteErrors = {
    // GEO satellites
    geo1: {
      '15min': { clock: -1.6003, x: 0.2691, y: -6.374, z: -2.071 },
      '30min': { clock: -2.6881, x: 0.3546, y: -10.4328, z: -2.8441},
      '1hour': { clock: -3.6016, x: 0.4405, y: -12.1432, z: -3.5941 },
      '2hour': { clock: 0.3889, x: 0.4412, y: 0.7324, z: 0.5145 }
    },
    // MEO satellites (GPS, Galileo)
    meo1: {
      '15min': { clock: 0.08, x: 0.06, y: 0.09, z: 0.12 },
      '30min': { clock: 0.15, x: 0.12, y: 0.18, z: 0.25 },
      '1hour': { clock: 0.30, x: 0.25, y: 0.35, z: 0.45 },
      '2hour': { clock: 0.60, x: 0.50, y: 0.65, z: 0.80 }
    },
    // LEO satellites
    meo2: {
      '15min': { clock: 0.05, x: 0.04, y: 0.06, z: 0.08 },
      '30min': { clock: 0.10, x: 0.08, y: 0.12, z: 0.15 },
      '1hour': { clock: 0.20, x: 0.16, y: 0.24, z: 0.30 },
      '2hour': { clock: 0.40, x: 0.32, y: 0.45, z: 0.55 }
    }
  };

  // Time intervals
  const timeIntervals = [
    { key: '15min', label: '0:15 hours' },
    { key: '30min', label: '0:30 hours' },
    { key: '1hour', label: '1:00 hour' },
    { key: '2hour', label: '2:00 hours' }
  ];

  // Get error values for the current satellite and time intervals
  const getTimeIntervalsWithErrors = () => {
    if (!satellite) {
      // Default to GEO if no satellite selected
      return timeIntervals.map(interval => ({
        ...interval,
        errors: satelliteErrors.geo1[interval.key]
      }));
    }

    // Try to match satellite type from ID
    const satId = satellite.id.toLowerCase();
    let errorSet;
    
    if (satId.includes('meo') || satId.includes('gps') || satId.includes('galileo')) {
      errorSet = satelliteErrors.meo1;
    } else if (satId.includes('meo2')) {
      errorSet = satelliteErrors.meo2;
    } else {
      // Default to GEO for all other cases
      errorSet = satelliteErrors.geo1;
    }

    return timeIntervals.map(interval => ({
      ...interval,
      errors: errorSet[interval.key]
    }));
  };

  const timeIntervalsWithErrors = getTimeIntervalsWithErrors();

  // Format error value with color coding
  const formatError = (value) => {
    // Determine color based on error magnitude
    let colorClass = 'text-green-400';
    if (value > 0.5|| value<-0.5) {
      colorClass = 'text-red-400';
    } else if (value > 0.3|| value<-0.3) {
      colorClass = 'text-yellow-400';
    }
    
    return (
      <span className={colorClass}>
        {value.toFixed(2)} m
      </span>
    );
  };

  return (
    <div className="w-full h-full flex flex-col">
      <h3 className="text-lg font-semibold text-hud-accent mb-3">
        8th Day Error Prediction
      </h3>
      
      {!satellite ? (
        <div className="flex-1 flex items-center justify-center text-gray-400">
          Select a satellite to view predictions
        </div>
      ) : (
        <div className="space-y-4 flex-1 overflow-y-auto">
          {timeIntervalsWithErrors.map(({ key, label, errors }) => (
            <div key={key} className="bg-hud-darker/50 p-3 rounded-lg border border-hud-border">
              <h4 className="text-sm font-medium text-gray-300 mb-2">
                {label} Prediction
              </h4>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="text-gray-400">Clock:</div>
                <div className="text-right">{formatError(errors.clock)}</div>
                
                <div className="text-gray-400 pl-2">• X:</div>
                <div className="text-right">{formatError(errors.x)}</div>
                
                <div className="text-gray-400 pl-2">• Y:</div>
                <div className="text-right">{formatError(errors.y)}</div>
                
                <div className="text-gray-400 pl-2">• Z:</div>
                <div className="text-right">{formatError(errors.z)}</div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      <div className="mt-3 text-xs text-gray-500 text-right">
        {satellite ? `Selected: ${satellite.id}` : 'No satellite selected'}
      </div>
    </div>
  );
};

export default PredictionPanel;