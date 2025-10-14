/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'hud-dark': '#0d1a26',
        'hud-darker': '#081018',
        'hud-accent': '#00f5ff',
        'hud-accent-alt': '#64ffda',
        'hud-green': '#00ff88',
        'hud-red': '#ff4757',
        'hud-border': '#1a3a52',
      },
      fontFamily: {
        'exo': ['"Exo 2"', 'sans-serif'],
        'roboto': ['Roboto', 'sans-serif'],
      },
      boxShadow: {
        'hud': '0 0 20px rgba(0, 245, 255, 0.3)',
        'hud-sm': '0 0 10px rgba(0, 245, 255, 0.2)',
      },
    },
  },
  plugins: [],
}
