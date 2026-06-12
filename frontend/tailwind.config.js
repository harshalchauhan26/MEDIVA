/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#152026",
        mint: "#2fbf9d",
        coral: "#f26d5b",
        clinic: "#eef8f5",
      },
      boxShadow: {
        panel: "0 20px 60px rgba(21, 32, 38, 0.12)",
      },
    },
  },
  plugins: [],
};
