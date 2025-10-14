# 🚀 G-ForcEST Quick Start Guide

## Installation (First Time Only)

### Step 1: Install Backend Dependencies
```bash
cd server
npm install
```

### Step 2: Install Frontend Dependencies
```bash
cd ../client
npm install
```

**OR** Install everything at once from root:
```bash
npm run install:all
```

---

## Running the Application

### Two Terminals Required

**Terminal 1 - Backend Server:**
```bash
cd server
npm run dev
```
✅ Backend running on **http://localhost:5000**

**Terminal 2 - Frontend:**
```bash
cd client
npm run dev
```
✅ Frontend running on **http://localhost:3000**

---

## 🌐 Access the Dashboard

Open your browser: **http://localhost:3000**

---

## 📊 What You'll See

1. **Top Row**: 4 KPI cards showing system metrics
2. **Center-Left**: Interactive world map with satellite markers
3. **Right Side**: 3 charts showing error data and trends

---

## 🎮 Interacting with the Dashboard

- **Click satellite markers** on the map to view detailed error data
- **Hover over charts** to see specific values
- **Dashboard auto-refreshes** every 30 seconds

---

## 🆘 Troubleshooting

**Backend won't start?**
- Make sure port 5000 is available
- Check if dependencies are installed

**Frontend shows error?**
- Ensure backend is running first
- Check if port 3000 is available

**Map not showing?**
- Check internet connection (map tiles load from CDN)
- Verify Leaflet CSS is loaded

---

## 📁 Project Structure

```
gforcest/
├── server/          # Backend (Port 5000)
├── client/          # Frontend (Port 3000)
└── README.md        # Full documentation
```

---

**That's it! Your G-ForcEST dashboard is ready! 🎉**

For detailed documentation, see [README.md](README.md)
