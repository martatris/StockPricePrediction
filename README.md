# Multi-Model Stock Price Prediction

This project predicts stock prices for major tech companies using multiple machine learning and deep learning models. The project compares the performance of six models and provides both numerical evaluation and visualizations.

## Tech Stocks Used:

* Apple (AAPL)
* Google (GOOG)
* Microsoft (MSFT)
* Amazon (AMZN)

## Models Implemented:

### Deep Learning Models:

* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)
* 1D CNN (Convolutional Neural Network)
  Traditional Machine Learning Models:
* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

## Features:

* Uses historical stock closing price data from Yahoo Finance.
* Predicts future stock prices using deep learning and traditional ML models.
* Prints final numeric results for each model:

  * Last-day actual vs predicted price
  * Predicted direction (increase/decrease)
  * Difference
  * RMSE and MAE
* Produces plots comparing actual vs predicted prices.
* Can be extended to other stocks or features.

## Dependencies:

* Python 3.9+
* Packages: yfinance, numpy, pandas, matplotlib, scikit-learn, tensorflow, xgboost
* Recommended: use a virtual environment or Conda to manage dependencies.

## Installation:

1. Clone the repository:
   git clone [https://github.com/yourusername/multi-model-stock-prediction.git](https://github.com/yourusername/multi-model-stock-prediction.git)
   cd multi-model-stock-prediction
2. Create a virtual environment (optional):
   python3 -m venv env
   source env/bin/activate   # macOS/Linux

   # env\Scripts\activate    # Windows
3. Install required packages:
   pip install -r requirements.txt

## Usage:

* Run the main script: python stock_prediction.py
* The script fetches historical data, trains all six models, prints final numeric predictions, and shows plots.

## Project Structure:

* stock_prediction.py  # Main script
* README.txt          # Project overview and instructions
* requirements.txt    # Python package dependencies

## Notes:

* Historical data is fetched using Yahoo Finance API.
* Deep learning models may take several minutes to train.
* Modify `tech_list` to add other stocks or change the date range.
