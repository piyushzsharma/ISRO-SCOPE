import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const ErrorHistory = ({ satellite }) => {
  // Generate mock error data for the last 7 days
  const errorData = useMemo(() => {
    if (!satellite) return [];
    
    const days = 7;
    return Array.from({ length: days }, (_, i) => {
      const date = new Date();
      date.setDate(date.getDate() - (days - 1 - i));
      return {
        date: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
        // Generate random error count between 0-10 for demo
        errors: Math.floor(Math.random() * 11),
      };
    });
  }, [satellite?.id]);

  return (
    <div className="h-full flex flex-col">
      <h3 className="text-lg font-semibold mb-3 text-white">
        {satellite ? `Error History - ${satellite.id}` : 'Satellite Error History'}
      </h3>
      <div className="flex-1">
        {!satellite ? (
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
            <p className="text-gray-400">Select a satellite from the 3D view to see its error history</p>
            <p className="text-sm text-gray-500 mt-2">
              Click on any satellite marker to view detailed error information
            </p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart
              data={errorData}
              margin={{ top: 5, right: 20, left: 0, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#2a4a6a" />
              <XAxis 
                dataKey="date" 
                tick={{ fill: '#a0aec0', fontSize: 12 }}
                axisLine={{ stroke: '#4a5568' }}
              />
              <YAxis 
                tick={{ fill: '#a0aec0', fontSize: 12 }}
                axisLine={{ stroke: '#4a5568' }}
                domain={[0, 10]}
                width={20}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1a365d', 
                  border: '1px solid #2c5282',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                }}
                itemStyle={{ color: '#e2e8f0' }}
                labelStyle={{ color: '#a0aec0' }}
              />
              <Line 
                type="monotone" 
                dataKey="errors" 
                name="Errors"
                stroke="#ff4757" 
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#ff4757' }}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
      <div className="mt-2 text-xs text-gray-400">
        {satellite ? 'Last 7 days of error occurrences' : 'No satellite selected'}
      </div>
    </div>
  );
};

export default ErrorHistory;
