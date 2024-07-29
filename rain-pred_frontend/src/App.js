
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [inputs, setInputs] = useState({ day1: '', day2: '', day3: '', day4: '' });
  const [predictedRainfall, setPredictedRainfall] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleChange = (e) => {
    setInputs({ ...inputs, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMessage(null);  // Reset error message on new submit

    // Ensure all input fields are filled
    if (!inputs.day1 || !inputs.day2 || !inputs.day3 || !inputs.day4) {
      setErrorMessage('Please enter values for all days.');
      return;
    }

    const rainfallData = [
      parseFloat(inputs.day1),
      parseFloat(inputs.day2),
      parseFloat(inputs.day3),
      parseFloat(inputs.day4),
    ];

    try {
      const response = await axios.post('http://localhost:8000/api/predict/', {
        rainfall: rainfallData,
      }, {
        headers: {
          'Content-Type': 'application/json',
        }
      });

      setPredictedRainfall(response.data.predicted_rainfall);
    } catch (error) {
      setErrorMessage('Error predicting rainfall. Please try again.');
      console.error('Error predicting rainfall:', error.response.data);  // Log the error for debugging
    }
  };

  return (
    <div className="App">
      <h1>Flood Prediction</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Day 1 Rainfall (mm):
          <input
            type="number"
            name="day1"
            value={inputs.day1}
            onChange={handleChange}
            required
          />
        </label>
        <br />
        <label>
          Day 2 Rainfall (mm):
          <input
            type="number"
            name="day2"
            value={inputs.day2}
            onChange={handleChange}
            required
          />
        </label>
        <br />
        <label>
          Day 3 Rainfall (mm):
          <input
            type="number"
            name="day3"
            value={inputs.day3}
            onChange={handleChange}
            required
          />
        </label>
        <br />
        <label>
          Day 4 Rainfall (mm):
          <input
            type="number"
            name="day4"
            value={inputs.day4}
            onChange={handleChange}
            required
          />
        </label>
        <br />
        <button type="submit">Predict Next Day Rainfall</button>
      </form>
      {predictedRainfall !== null && (
        <div>
          <h2>Predicted Rainfall: {predictedRainfall.toFixed(2)} mm</h2>
        </div>
      )}
      {
        predictedRainfall !==null && 
          predictedRainfall.toFixed(2) > 80 && (
            <div>
              <h2>There is a High Chance of Flood</h2>
            </div>
          )
        
      }
      {errorMessage && (
        <div style={{ color: 'red' }}>
          <h2>Error: {errorMessage}</h2>
        </div>
      )}
    </div>
  );
}

export default App;

