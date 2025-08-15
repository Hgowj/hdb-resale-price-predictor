# 🏠 Singapore HDB Resale Price Predictor

A machine learning web application that predicts Singapore HDB (Housing Development Board) resale flat prices using Random Forest regression.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

![Main Interface](https://github.com/user-attachments/assets/your-main-interface-image-id)
*Clean, intuitive interface for inputting flat details*

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

1. **Clone the repository**
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Add your data:** Place the HDB resale CSV file in the project folder
4. **Train the model:** `python src/train_model.py`
5. **Run the app:** `streamlit run src/app.py`

## 📦 Installation

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

![Prediction Results](https://github.com/user-attachments/assets/5c1e8bfd-e882-4c45-890e-7933b9f3a8bf)
*Example prediction showing price estimate with confidence interval*

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

### Quick Examples
The app includes pre-filled examples for common scenarios:
- **Budget Option**: 3-room flat in mature estate
- **Family Home**: 4-room flat in established town
- **Premium Choice**: 5-room flat in prime location

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
├── screenshots/
│   ├── main-interface.png     # App interface screenshot
│   ├── prediction-example.png # Prediction results screenshot
│   └── feature-importance.png # Feature analysis chart
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

![Feature Importance](https://github.com/user-attachments/assets/0aa331e9-4f11-4c73-a4f5-87ad6580b63f)

*Analysis showing which factors most influence HDB prices*

1. **Remaining Lease** (47%) - Longer leases = higher value
2. **Flat Model** (20%) - Design/layout has significant impact  
3. **Floor Area** (13%) - Size is an important price factor
4. **Storey Range** (12%) - Higher floors command premium
5. **Flat Type** (5%) - Room count has moderate impact
6. **Town** (3%) - Location has smaller but notable effect

### Data Preprocessing
- **Categorical Encoding**: Label encoding for towns, flat types, models
- **Feature Engineering**: Lease remaining calculation from lease commencement
- **Data Filtering**: Focus on 2020-2025 for current market relevance
- **Outlier Handling**: Statistical filtering for data quality

## 📊 Data Source

To use this predictor:

1. **Download HDB resale data** from [data.gov.sg](https://data.gov.sg/dataset/resale-flat-prices)
2. **Place the CSV file** in your project root folder
3. **Run training script**: `python src/train_model.py`
4. **Launch app**: `streamlit run src/app.py`

**Expected file:** `ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv`

### Data Fields Used
- `month`: Transaction month
- `town`: HDB town/estate
- `flat_type`: Number of rooms (1-5 ROOM, EXECUTIVE)
- `storey_range`: Floor range (e.g., "04 TO 06")
- `floor_area_sqm`: Floor area in square meters
- `flat_model`: Flat design model
- `lease_commence_date`: Year lease started
- `resale_price`: Transaction price (target variable)

## 🎯 Use Cases

- **Home Buyers**: Estimate fair market price before viewing
- **Sellers**: Price flats competitively for quick sale
- **Investors**: Analyze market trends and opportunities
- **Researchers**: Study Singapore housing market dynamics
- **Real Estate Agents**: Provide data-driven price guidance

## 🚀 Future Enhancements

- [ ] **Advanced Features**: Add nearby amenities (MRT, schools, malls)
- [ ] **Time Series**: Incorporate market trend predictions
- [ ] **Comparison Tool**: Side-by-side flat comparisons
- [ ] **Market Analysis**: Town-level price trend charts
- [ ] **Export Options**: Save predictions as PDF reports
- [ ] **Mobile App**: React Native mobile version

## ⚠️ Disclaimer

This predictor provides estimates based on historical data and should be used as a reference only. Actual market prices may vary due to:

- Current market conditions and sentiment
- Specific flat characteristics not captured in data
- Economic factors and government policies
- Individual buyer/seller circumstances
- Renovation status and flat condition
- Nearby developments and future plans

**Always consult qualified real estate professionals for major decisions.**

## 🧪 Model Validation

The model was validated using:
- **Cross-validation**: 5-fold CV with consistent performance
- **Train/Test Split**: 80/20 split with temporal validation
- **Error Analysis**: Residual analysis across price ranges
- **Feature Stability**: Importance rankings consistent across folds

### Performance by Flat Type
| Flat Type | MAE (SGD) | R² Score | Sample Size |
|-----------|-----------|----------|-------------|
| 3 ROOM    | $38,450   | 0.862    | 32,156      |
| 4 ROOM    | $45,230   | 0.881    | 58,942      |
| 5 ROOM    | $52,180   | 0.875    | 41,238      |
| EXECUTIVE | $61,420   | 0.845    | 16,650      |

## 📈 Performance Benchmarks

Compared to simple baseline models:
- **Linear Regression**: 0.742 R² (vs 0.874 Random Forest)
- **Average Price**: $120k MAE (vs $46k Random Forest)
- **Location Only**: 0.651 R² (shows importance of multiple features)

## 🛠️ Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
plotly>=5.15.0
```

## 🙏 Acknowledgments

- **Singapore Housing Development Board** for providing comprehensive open data
- **Streamlit team** for the amazing web framework that makes deployment simple
- **scikit-learn community** for robust machine learning tools
- **Singapore open data initiative** for promoting data transparency

⭐ **Star this repo if you found it helpful!** ⭐

*Built with ❤️ for the Singapore housing community*
