import { useState, useEffect } from 'react';
import KPICards from './components/KPICards';
import GlobeView from './components/GlobeView';
import ErrorHistory from './components/ErrorHistory';
import Charts from './components/Charts';
import './App.css';

function App() {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('Global Overview');
  const [selectedSatellite, setSelectedSatellite] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredSatellites, setFilteredSatellites] = useState([]);

  useEffect(() => {
    if (dashboardData?.satellite_positions) {
      const filtered = dashboardData.satellite_positions.filter(satellite => 
        satellite.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
        satellite.type.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredSatellites(filtered);
    }
  }, [searchTerm, dashboardData]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const apiUrl = import.meta.env.DEV 
          ? 'http://localhost:5000/api/dashboard-data' 
          : '/api/dashboard-data';
        
        const response = await fetch(apiUrl);
        if (!response.ok) {
          throw new Error('Failed to fetch dashboard data');
        }
        const data = await response.json();
        setDashboardData(data);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError(err.message);
        setLoading(false);
      }
    };

    fetchData();
    
    // Refresh data every 30 seconds
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-hud-dark flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-hud-accent mb-4"></div>
          <p className="text-hud-accent text-xl font-exo">Loading G-ForcEST Dashboard...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-hud-dark flex items-center justify-center">
        <div className="text-center">
          <p className="text-hud-red text-xl font-exo mb-4">Error: {error}</p>
          <p className="text-gray-400">Please ensure the backend server is running on port 5000</p>
        </div>
      </div>
    );
  }

  const tabs = ['Global Overview', 'Satellite Deep Dive', 'Model Performance', 'System & Configuration'];

  return (
    <div className="min-h-screen bg-hud-dark">
      {/* Header */}
      <header className="bg-hud-darker border-b border-hud-border px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-hud-accent to-hud-green flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3.055 11H5a2 2 0 012 2v1a2 2 0 002 2 2 2 0 012 2v2.945M8 3.935V5.5A2.5 2.5 0 0010.5 8h.5a2 2 0 012 2 2 2 0 104 0 2 2 0 012-2h1.064M15 20.488V18a2 2 0 012-2h3.064M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h1 className="text-2xl font-bold text-hud-accent font-exo tracking-wider">
              G-ForcAST
            </h1>
          </div>
          <div className="flex items-center space-x-4">
            <div className="relative">
              <input
                type="text"
                placeholder="Search satellites..."
                className="bg-hud-darker border border-hud-border rounded-lg py-1 px-3 pl-10 text-white focus:outline-none focus:ring-2 focus:ring-hud-accent w-64"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <svg 
                className="w-5 h-5 text-gray-400 absolute left-2.5 top-1/2 transform -translate-y-1/2" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <button className="p-2 hover:bg-hud-border rounded-lg transition-colors">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
              </svg>
            </button>
            <button className="p-2 hover:bg-hud-border rounded-lg transition-colors">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-hud-darker border-b border-hud-border px-6">
        <div className="flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`py-4 px-2 font-medium text-sm transition-colors relative ${
                activeTab === tab
                  ? 'text-white'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              {tab}
              {activeTab === tab && (
                <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-hud-accent"></div>
              )}
            </button>
          ))}
        </div>
      </nav>

      {/* Main Content */}
      <div className="p-6">
        {/* KPI Cards */}
        <KPICards data={dashboardData?.kpi_data} />

        {/* Main Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          {/* Left Column - Globe */}
          <div className="lg:col-span-2">
            <div className="h-full flex flex-col">
              <div className="mb-2 text-sm text-gray-400">
                {searchTerm && (
                  <span>Showing {filteredSatellites.length} satellite{filteredSatellites.length !== 1 ? 's' : ''} matching "{searchTerm}"</span>
                )}
              </div>
              <div className="flex-1">
                <GlobeView 
                  satellites={searchTerm ? filteredSatellites : dashboardData?.satellite_positions}
                  selectedSatellite={selectedSatellite}
                  onSelectSatellite={setSelectedSatellite}
                />
              </div>
            </div>
          </div>

          {/* Right Column - Error History and Charts */}
          <div className="space-y-6">
            {/* Error History Card */}
            <div className="bg-hud-darker border border-hud-border rounded-lg p-4 h-80">
              <ErrorHistory satellite={selectedSatellite} />
            </div>
            
            {/* Charts Section */}
            <div className="bg-hud-darker border border-hud-border rounded-lg p-4">
              <Charts data={dashboardData?.charts_data} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
