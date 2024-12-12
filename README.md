# LSTM-Predict the opening price of a stock

## Overview

This repository contains a Python script (`predict.py`) designed to predict the opening price of a stock using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN). The model is trained on historical stock price data and is capable of forecasting future prices based on past trends.

## Features

- **Data Preprocessing**: The script includes functionality to load, preprocess, and standardize stock data.
- **LSTM Model**: A sequential LSTM model is built with dropout layers to prevent overfitting.
- **Training and Testing**: The dataset is split into training and testing sets to evaluate the model's performance.
- **Performance Metrics**: The script calculates and displays key performance metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² (Coefficient of Determination).
- **Visualization**: The script includes code to visualize both the performance metrics and the predicted versus actual stock prices.

## Getting Started

### Prerequisites

- Python 3.11
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

### Installation

1. Clone the repository or download the `predict.py` file.
2. Ensure you have the required libraries installed. You can install them using pip:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

3. Place your stock data CSV file in the same directory as the `predict.py` script, or update the file path in the script to match your data file's location.

### Usage

1. Run the `predict.py` script using Python:

```bash
python predict.py
```

2. The script will execute and produce two plots:
   - One showing the evaluation metrics for the model.
   - Another showing the actual and predicted stock prices.

## Code Breakdown

### Data Loading and Preprocessing

The script starts by loading the stock data from a CSV file, ensuring the 'trade_time' column is in datetime format, and setting it as the index. It then selects the 'open' column, which represents the opening price, and scales the data using MinMaxScaler.

### Creating Training and Testing Data

The script creates sequences of 500 days of data to predict the next day's opening price. It reshapes the data to fit the LSTM model's input requirements.

### Model Architecture

The LSTM model consists of two LSTM layers with 50 units each, followed by dropout layers to reduce overfitting. The final layer is a dense layer with a single unit for the price prediction.

### Training the Model

The model is trained on the training dataset for 10 epochs with a batch size of 32.

### Prediction and Evaluation

After training, the model makes predictions on the test set. The predictions and actual values are then inverse-transformed to their original scale. The script calculates and prints the performance metrics.

### Visualization

The script generates two plots:

- A bar chart displaying the evaluation metrics.
- A line chart comparing the actual and predicted stock prices.

## Performance

The performance of the model can be evaluated by looking at the printed metrics and the visualizations. Lower values for MSE, RMSE, and MAE, and a higher R² value indicate better performance.

## Contributing

Feel free to contribute to this project by submitting pull requests or suggesting improvements.

## License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).

