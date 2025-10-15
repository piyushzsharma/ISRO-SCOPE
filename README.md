# SCOPE Dashboard
## Satellite Clock and Orbit Prediction Error Analysis

A professional, futuristic 2D dashboard for visualizing GNSS satellite error prediction data with a Heads-Up Display (HUD) aesthetic.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 Features

- **Real-time KPI Monitoring**: Four key performance indicator cards showing system accuracy, error counts, model status, and last update time
- **Interactive 3D Globe**: Rotating Earth with satellite markers in orbital rings, clickable for detailed information
- **Navigation Tabs**: Multi-section dashboard with Global Overview, Satellite Deep Dive, Model Performance, and System Configuration
- **Data Visualization**: Three professional charts showing error distribution, bell-curve histograms, and trend analysis
- **Futuristic HUD Design**: Dark theme with glowing teal/green accents, inspired by advanced monitoring systems
- **Responsive Layout**: Fully responsive design that works on all screen sizes
- **Auto-refresh**: Dashboard automatically updates every 30 seconds

---

## 🛠️ Technology Stack

### Frontend
- **React 18** - Modern UI library
- **Vite** - Fast build tool and development server
- **Tailwind CSS** - Utility-first CSS framework
- **Three.js** - 3D graphics library
- **React Three Fiber** - React renderer for Three.js
- **React Three Drei** - Useful helpers for 3D components
- **Chart.js** - Professional charting library
- **React Chart.js 2** - React wrapper for Chart.js

### Backend
- **Node.js** - JavaScript runtime
- **Express.js** - Web application framework
- **CORS** - Cross-Origin Resource Sharing middleware

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v16.0.0 or higher)
- **npm** (v7.0.0 or higher)

---

## 🚀 Installation & Setup

### Step 1: Navigate to Project Directory

```bash
cd gforcest
```

### Step 2: Install Backend Dependencies

```bash
cd server
npm install
```

### Step 3: Install Frontend Dependencies

```bash
cd ../client
npm install
```

---

## ▶️ Running the Application

### Option 1: Run Both Servers (Recommended for Development)

You'll need **two terminal windows**.

#### Terminal 1 - Backend Server:
```bash
cd server
npm run dev
```
✅ Backend will start on **http://localhost:5000**

#### Terminal 2 - Frontend Development Server:
```bash
cd client
npm run dev
```
✅ Frontend will start on **http://localhost:3000**

### Option 2: Production Mode

#### Start Backend:
```bash
cd server
npm start
```

#### Build Frontend:
```bash
cd client
npm run build
```

---

## 🌐 Accessing the Dashboard

Once both servers are running:
1. Open your web browser
2. Navigate to **http://localhost:3000**
3. The G-ForcEST dashboard will load with all visualizations

---

## 📊 Dashboard Components

### 1. KPI Cards (Top Row)
Four key performance indicators:
- **Overall System Accuracy**: Current system accuracy in meters
- **Satellites with High Error**: Count of satellites experiencing high errors
- **Model Status**: Current prediction model status
- **Last Data Update**: Timestamp of the most recent data update

### 2. Interactive 3D Globe (Left Side)
- **Rotating 3D Earth** with realistic rendering
- **Orbital rings** showing satellite paths
- **Satellite markers** color-coded by status:
  - 🟢 **Green**: Nominal operation
  - 🔴 **Red**: High error detected
- **Click markers** to view detailed satellite information card:
  - Satellite ID and type
  - Current status
  - Ephemeris error
  - Clock error
- **Interactive controls**: Rotate, zoom, and pan the globe

### 3. Charts (Right Column)

#### Chart 1: Error Distribution by Constellation
Bar chart showing error distribution across different satellite constellations (MEO, GEO/GSO).

#### Chart 2: Live Prediction Error Histogram
Histogram displaying the distribution of 8th day residual errors.

#### Chart 3: Predicted Error Trend
Line chart showing the predicted error trend over time for all satellites.

---

## 📁 Project Structure

```
gforcest/
├── client/                      # Frontend React application
│   ├── src/
│   │   ├── components/          # React components
│   │   │   ├── KPICards.jsx     # KPI indicator cards
│   │   │   ├── MapContainer.jsx # Interactive map
│   │   │   └── Charts.jsx       # Chart visualizations
│   │   ├── App.jsx              # Main application component
│   │   ├── App.css              # App-specific styles
│   │   ├── main.jsx             # Application entry point
│   │   └── index.css            # Global styles
│   ├── index.html               # HTML entry point
│   ├── vite.config.js           # Vite configuration
│   ├── tailwind.config.js       # Tailwind CSS configuration
│   ├── postcss.config.js        # PostCSS configuration
│   └── package.json             # Frontend dependencies
│
├── server/                      # Backend Express server
│   ├── index.js                 # Server entry point & API
│   └── package.json             # Backend dependencies
│
└── README.md                    # This file
```

---

## 🔌 API Documentation

### GET `/api/dashboard-data`

Returns all dashboard data in a single JSON response.

**Response Format:**
```json
{
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
      "error_data": {
        "ephemeris_error": "0.45m",
        "clock_error": "0.8ns"
      }
    }
  ],
  "charts_data": {
    "constellation_distribution": {
      "labels": ["MEO", "MEO", "MEO", "GEO/GSO", "GEO/GSO"],
      "values": [25, 60, 45, 85, 75]
    },
    "error_histogram": {
      "labels": ["-2m", "-1m", "0m", "1m", "2m"],
      "values": [15, 45, 85, 40, 10]
    },
    "error_trend": {
      "labels": ["-1h", "0h", "1h", "2h", "4h"],
      "values": [0.1, 0.12, 0.18, 0.25, 0.4]
    }
  }
}
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "OK",
  "timestamp": "2024-03-08T12:00:00.000Z"
}
```

---

## 🎨 Customization

### Changing Colors

Edit `client/tailwind.config.js`:

```javascript
colors: {
  'hud-dark': '#0d1a26',        // Main background
  'hud-darker': '#081018',      // Card backgrounds
  'hud-accent': '#00f5ff',      // Primary accent (teal)
  'hud-accent-alt': '#64ffda',  // Secondary accent
  'hud-green': '#00ff88',       // Success/nominal
  'hud-red': '#ff4757',         // Error/warning
  'hud-border': '#1a3a52',      // Border color
}
```

### Adding More Satellites

Edit `server/index.js` and add new satellite objects to the `satellite_positions` array:

```javascript
{
  "id": "MEO-04",
  "type": "MEO",
  "position": [latitude, longitude],
  "status_color": "green",
  "error_data": {
    "ephemeris_error": "0.35m",
    "clock_error": "0.7ns"
  }
}
```

### Modifying Chart Data

Edit the `charts_data` object in `server/index.js` to update chart values.

---

## 🐛 Troubleshooting

### Port Already in Use

**Backend (Port 5000):**
- Change the port in `server/index.js`:
  ```javascript
  const PORT = process.env.PORT || 5001;
  ```

**Frontend (Port 3000):**
- Change the port in `client/vite.config.js`:
  ```javascript
  server: {
    port: 3001,
  }
  ```

### CORS Errors

Ensure the backend server is running and CORS is enabled in `server/index.js`.

### Map Not Displaying

1. Check browser console for errors
2. Ensure Leaflet CSS is loaded in `index.html`
3. Verify internet connection (map tiles are loaded from CDN)

### Charts Not Rendering

1. Ensure Chart.js is properly installed
2. Check browser console for errors
3. Verify data format matches expected structure

---

## 📦 Building for Production

### Build Frontend

```bash
cd client
npm run build
```

The optimized production build will be in `client/dist/`.

### Deploy

You can serve the production build using:
- Any static file server
- Integration with Express backend
- Cloud platforms (Vercel, Netlify, etc.)

---

## 🔧 Development Tips

### Auto-refresh on Changes

Both servers support hot-reload:
- **Backend**: Uses nodemon to restart on file changes
- **Frontend**: Vite provides instant HMR (Hot Module Replacement)

### Debugging

- Check browser console for frontend errors
- Check terminal output for backend errors
- Use React DevTools for component inspection
- Use Network tab to monitor API calls

---

## 📈 Future Enhancements

- [ ] Real-time data streaming via WebSockets
- [ ] Historical data playback
- [ ] Export data to CSV/JSON
- [ ] User authentication and role-based access
- [ ] Custom alert thresholds
- [ ] 3D satellite orbit visualization
- [ ] Multi-language support
- [ ] Dark/Light theme toggle

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 👥 Authors

- **ISRO Development Team**

---

## 🙏 Acknowledgments

- React.js team for the amazing framework
- Leaflet for the powerful mapping library
- Chart.js for beautiful data visualizations
- Tailwind CSS for the utility-first approach

---

## 📞 Support

For issues and questions:
- Open an issue in the repository
- Contact the development team

---

**Built with ❤️ for GNSS Error Analysis and Prediction**
