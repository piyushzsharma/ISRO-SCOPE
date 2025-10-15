import React from 'react';

const KPICards = ({ data }) => {
  if (!data) return null;

  const cards = [
    {
      title: 'Model Status',
      value: data.model_status,
      unit: '',
      color: 'text-hud-green',
      badge: true,
    },
    {
      title: 'Satellites with High Error',
      value: data.high_error_count,
      unit: '',
      color: 'text-white',
    },
    
    {
      title: 'Last Data Update',
      value: data.last_update,
      unit: '',
      color: 'text-white',
      small: true,
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {cards.map((card, index) => (
        <div
          key={index}
          className="bg-hud-darker border border-hud-border rounded-lg p-5"
        >
          <h3 className="text-sm text-gray-400 mb-3 font-exo">
            {card.title}
          </h3>
          
          {card.badge ? (
            <div className="inline-flex items-center px-3 py-1.5 rounded bg-hud-green/20 border border-hud-green">
              <svg className="w-4 h-4 text-hud-green mr-2" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="text-hud-green text-xs font-semibold">{card.value}</span>
            </div>
          ) : (
            <div className={`${card.color} font-bold ${card.small ? 'text-base' : 'text-3xl'}`}>
              {card.value} {card.unit && <span className="text-sm text-gray-400 font-normal">{card.unit}</span>}
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

export default KPICards;
