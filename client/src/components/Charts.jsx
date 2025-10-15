import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar, Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  Title,
  Tooltip,
  Legend
);

const Charts = ({ data }) => {
  if (!data) return null;

  // Common chart options
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        backgroundColor: 'rgba(13, 26, 38, 0.95)',
        titleColor: '#00f5ff',
        bodyColor: '#ffffff',
        borderColor: '#00f5ff',
        borderWidth: 1,
        padding: 12,
        titleFont: {
          family: 'Exo 2',
          size: 14,
        },
        bodyFont: {
          family: 'Exo 2',
          size: 12,
        },
      },
    },
    scales: {
      x: {
        grid: {
          color: 'rgba(26, 58, 82, 0.2)',
          borderColor: 'transparent',
        },
        ticks: {
          color: '#64ffda',
          font: {
            family: 'Exo 2',
            size: 10,
          },
        },
      },
      y: {
        grid: {
          color: 'rgba(26, 58, 82, 0.2)',
          borderColor: 'transparent',
        },
        ticks: {
          color: '#64ffda',
          font: {
            family: 'Exo 2',
            size: 10,
          },
        },
      },
    },
  };

  // Bar Chart 1: Constellation Distribution
  const constellationData = {
    labels: data.constellation_distribution.labels,
    datasets: [
      {
        label: 'Error Distribution',
        data: data.constellation_distribution.values,
        backgroundColor: ['#00bfff', '#00bfff', '#00bfff', '#00ff88', '#00ff88'],
        borderColor: 'transparent',
        borderWidth: 0,
        borderRadius: 4,
      },
    ],
  };

  // Bar Chart 2: Error Histogram (Bell Curve)
  const histogramData = {
    labels: data.error_histogram.labels,
    datasets: [
      {
        label: 'Frequency',
        data: data.error_histogram.values,
        backgroundColor: '#00bfff',
        borderColor: 'transparent',
        borderWidth: 0,
        borderRadius: 2,
      },
    ],
  };

  // Line Chart: Error Trend
  const trendData = {
    labels: data.error_trend.labels,
    datasets: [
      {
        label: 'Predicted Error',
        data: data.error_trend.values,
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        borderWidth: 2,
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#00ff88',
        pointBorderColor: '#00ff88',
        pointRadius: 0,
        pointHoverRadius: 4,
      },
    ],
  };

  return (
    <div className="space-y-4">
      {/* Chart 1: Constellation Distribution */}
      <div className="bg-hud-darker border border-hud-border rounded-lg p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-white font-exo">
            Error Distribution by Constellation
          </h3>
        </div>
        <div className="h-[140px]">
          <Bar data={constellationData} options={commonOptions} />
        </div>
      </div>
    </div>
  );
};

export default Charts;
