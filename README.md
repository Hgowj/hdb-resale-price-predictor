# hdb-resale-price-predictor
ML web app for predicting Singapore HDB resale prices

# 🏠 Singapore HDB Resale Price Predictor

A machine learning web application that predicts Singapore HDB (Housing Development Board) resale flat prices using Random Forest regression.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🎯 Features

- **Accurate Predictions**: 87.4% R² score with ~$46k average error
- **Recent Data**: Trained on 2020-2025 HDB transaction data (148k+ samples)
- **Interactive Web App**: User-friendly Streamlit interface
- **Real-time Predictions**: Instant price estimates with confidence ranges
- **Feature Analysis**: Understand what factors influence pricing most

## 📊 Model Performance

- **Mean Absolute Error (MAE)**: $46,380
- **R² Score**: 0.874
- **Training Samples**: 148,986 transactions
- **Data Period**: 2020-2025 (post-pandemic market data)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/hdb-price-predictor.git
   cd hdb-price-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (optional - pre-trained model included)
   ```bash
   python src/train_model.py
   ```

4. **Run the web application**
   ```bash
   streamlit run src/app.py
   ```

5. **Open your browser** to `http://localhost:8501`

## 🎮 Usage

### Web Interface
1. Enter flat details (town, type, floor area, etc.)
2. Click "Predict Price" 
3. Get instant price estimate with confidence range
4. Use quick examples for common flat types

### Example Prediction
```
Input:
- Town: TAMPINES
- Flat Type: 4 ROOM
- Storey Range: 07 TO 09
- Floor Area: 95 sqm
- Flat Model: Model A
- Remaining Lease: 85 years

Output: ~$520,000 (Range: $468k - $572k)
```

## 📁 Project Structure

```
hdb-price-predictor/
├── src/
│   ├── app.py                 # Streamlit web application
│   └── train_model.py         # Model training script
├── models/
│   └── flat_price_model_2020_2025.pkl  # Trained model
├── data/
│   └── ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv
├── requirements.txt           # Python dependencies
└── README.md                 # Project documentation
```

## 🔧 Technical Details

### Features Used
- **Location**: Town/neighborhood
- **Property**: Flat type, storey range, floor area
- **Age**: Remaining lease duration
- **Design**: Flat model/layout

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Trees**: 50 estimators
- **Max Depth**: 20 levels
- **Regularization**: Min samples split/leaf constraints

### Feature Importance
1. **Floor Area** (47%) - Size is the biggest price factor
2. **Town** (20%) - Location premium matters significantly  
3. **Storey Range** (13%) - Higher floors command premium
4. **Remaining Lease** (12%) - Longer leases = higher value
5. **Flat Model** (5%) - Design/layout has moderate impact
6. **Flat Type** (3%) - Type less important than actual size

## 📈 Data Source

- **Dataset**: Singapore HDB Resale Transaction Data
- **Period**: January 2020 - 2025
- **Size**: 148,986 transactions
- **Source**: Singapore government open data
- **Features**: Town, flat type, storey range, floor area, flat model, remaining lease

## 🎯 Use Cases

- **Home Buyers**: Estimate fair market price before viewing
- **Sellers**: Price flats competitively for quick sale
- **Investors**: Analyze market trends and opportunities
- **Researchers**: Study Singapore housing market dynamics

## ⚠️ Disclaimer

This predictor provides estimates based on historical data and should be used as a reference only. Actual market prices may vary due to:
- Current market conditions
- Specific flat characteristics
- Economic factors
- Individual buyer/seller circumstances

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Singapore Housing Development Board for providing open data
- Streamlit team for the amazing web framework
- scikit-learn community for machine learning tools

## 📞 Contact

Project Link: [https://github.com/yourusername/hdb-price-predictor](https://github.com/yourusername/hdb-price-predictor)

---

⭐ **Star this repo if you found it helpful!** ⭐
