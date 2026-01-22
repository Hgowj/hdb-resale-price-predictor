# HDB Resale Price Predictor

A web application that predicts HDB resale flat prices in Singapore using machine learning.

<img width="1863" height="941" alt="image" src="https://github.com/user-attachments/assets/cd9364cc-96b4-427a-8940-397d94c23ccf" />

## About

This application uses a Random Forest model trained on 148,000+ recent HDB resale transactions (2020-2025) to estimate flat prices based on location, flat type, floor area, remaining lease, and other factors.

## Features

- Price predictions for all 26 towns and 7 flat types in Singapore
- Support for 21 different flat models
- Interactive interface with price range estimates
- Feature importance visualization
- Pre-configured examples for quick testing

## Tech Stack

- **Frontend**: Streamlit
- **Machine Learning**: scikit-learn (Random Forest Regressor)
- **Data Processing**: pandas, numpy
- **Visualization**: plotly

## Running Locally

1. Clone the repository
```bash
git clone https://github.com/Hgowj/hdb-resale-price-predictor.git
cd hdb-resale-price-predictor
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the application
```bash
streamlit run src/app.py
```

## Data Source

HDB resale price data from data.gov.sg (January 2017 onwards)

## Disclaimer

This prediction is based on historical data and should be used as a reference only. Actual market prices may vary due to market conditions and specific flat characteristics.
