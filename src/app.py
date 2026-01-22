"""
HDB Resale Price Predictor - Streamlit Web Application
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    """Load the trained model with caching for performance"""
    model_path = 'models/flat_price_model_2020_2025.pkl'
    
    if not os.path.exists(model_path):
        st.error("❌ Model not found! Please run `python train_model.py` first to train the model.")
        st.stop()
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

def predict_price(model_data, user_input):
    """Make prediction based on user input"""
    model = model_data['model']
    label_encoders = model_data['label_encoders']
    features = model_data['features']
    
    # Create input dataframe
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical variables
    categorical_features = ['town', 'flat_type', 'storey_range', 'flat_model']
    warnings = []
    
    for feature in categorical_features:
        if feature in input_df.columns and feature in label_encoders:
            try:
                input_df[feature] = label_encoders[feature].transform(input_df[feature])
            except ValueError:
                warnings.append(f"Unknown {feature}: '{user_input[feature]}'")
                # Use the most common encoded value (mode)
                input_df[feature] = 0
    
    # Make prediction
    predicted_price = model.predict(input_df[features])[0]
    
    return predicted_price, warnings

def main():
    # Load model data
    model_data = load_model()
    
    # Header
    st.title("🏠 Singapore HDB Resale Price Predictor")
    st.markdown("---")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        
        if 'performance' in model_data:
            perf = model_data['performance']
            st.metric("Validation MAE", f"${perf['validation_mae']:,.0f}")
            st.metric("R² Score", f"{perf['validation_r2']:.3f}")
            st.metric("Training Samples", f"{perf['training_samples']:,}")
        
        st.markdown("---")
        st.header("🎯 How to Use")
        st.markdown("""
        1. Enter the flat details in the form
        2. Click 'Predict Price'
        3. View your estimated price range
        4. Check the feature importance chart
        """)
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        This predictor uses a Random Forest model trained on recent HDB resale data from 2020-2025.
        
        **Coverage:** All 26 towns, 7 flat types, 21 flat models
        **Accuracy:** Based on 148k+ recent transactions
        **Features:** Location, size, type, age, etc.
        **Data Period:** 2020-2025 (Most Recent)
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🔧 Enter Flat Details")
        
        # Quick examples (moved before form)
        st.subheader("💡 Quick Examples")
        example_col1, example_col2 = st.columns(2)
        
        # Initialize session state for examples
        if 'example_selected' not in st.session_state:
            st.session_state.example_selected = None
        
        with example_col1:
            if st.button("🏠 Typical 4-Room", use_container_width=True):
                st.session_state.example_selected = 'typical'
        
        with example_col2:
            if st.button("🏢 Executive Flat", use_container_width=True):
                st.session_state.example_selected = 'executive'
        
        st.info("💡 **Tip:** The app now includes all 26 towns and 21 flat models from the official dataset!")
        
        # Additional quick examples
        example_col3, example_col4 = st.columns(2)
        
        with example_col3:
            if st.button("🏙️ Central Area", use_container_width=True):
                st.session_state.example_selected = 'central'
        
        with example_col4:
            if st.button("🌊 Marine Parade", use_container_width=True):
                st.session_state.example_selected = 'marine'
        
        # Set default values based on example selection
        if st.session_state.example_selected == 'typical':
            default_town = 'TAMPINES'
            default_flat_type = '4 ROOM'
            default_storey = '07 TO 09'
            default_area = 95.0
            default_model = 'Model A'
            default_lease = 85.0
        elif st.session_state.example_selected == 'executive':
            default_town = 'CLEMENTI'
            default_flat_type = 'EXECUTIVE'
            default_storey = '10 TO 12'
            default_area = 130.0
            default_model = 'Premium Apartment'
            default_lease = 75.0
        elif st.session_state.example_selected == 'central':
            default_town = 'CENTRAL AREA'
            default_flat_type = '3 ROOM'
            default_storey = '04 TO 06'
            default_area = 85.0
            default_model = 'Improved'
            default_lease = 70.0
        elif st.session_state.example_selected == 'marine':
            default_town = 'MARINE PARADE'
            default_flat_type = '5 ROOM'
            default_storey = '13 TO 15'
            default_area = 115.0
            default_model = 'Premium Apartment'
            default_lease = 80.0
        else:
            default_town = 'TAMPINES'
            default_flat_type = '4 ROOM'
            default_storey = '07 TO 09'
            default_area = 95.0
            default_model = 'Model A'
            default_lease = 85.0
        
        # Create form for better UX
        with st.form("prediction_form"):
            # All available towns from the dataset (26 towns total)
            all_towns = [
                'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 
                'BUKIT PANJANG', 'BUKIT TIMAH', 'CENTRAL AREA', 'CHOA CHU KANG', 
                'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST', 
                'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 
                'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG', 'SERANGOON', 'TAMPINES', 
                'TOA PAYOH', 'WOODLANDS', 'YISHUN'
            ]
            
            # Get index for default town
            town_index = all_towns.index(default_town) if default_town in all_towns else 0
            
            town = st.selectbox(
                "🏘️ Town",
                options=all_towns,
                index=town_index,
                help="Select the town where the flat is located"
            )
            
            # All available flat types from dataset
            flat_type_options = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
            flat_type_index = flat_type_options.index(default_flat_type) if default_flat_type in flat_type_options else 3  # Default to 4 ROOM
            
            flat_type = st.selectbox(
                "🏠 Flat Type",
                options=flat_type_options,
                index=flat_type_index,
                help="Select the type of flat"
            )
            
            # All available storey ranges from dataset
            storey_options = [
                '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', 
                '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30', 
                '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', 
                '46 TO 48', '49 TO 51'
            ]
            storey_index = storey_options.index(default_storey) if default_storey in storey_options else 2  # Default to 07 TO 09
            
            storey_range = st.selectbox(
                "🏢 Storey Range",
                options=storey_options,
                index=storey_index,
                help="Select the storey range of the flat"
            )
            
            # Two columns for numerical inputs
            num_col1, num_col2 = st.columns(2)
            
            with num_col1:
                floor_area = st.number_input(
                    "📐 Floor Area (sqm)",
                    min_value=20.0,
                    max_value=280.0,  # Updated based on data range
                    value=default_area,
                    step=1.0,
                    help="Enter the floor area in square meters"
                )
            
            with num_col2:
                remaining_lease = st.number_input(
                    "📅 Remaining Lease (years)",
                    min_value=40.0,
                    max_value=99.0,
                    value=default_lease,
                    step=0.5,
                    help="Enter the remaining lease in years"
                )
            
            # All available flat models from dataset
            model_options = [
                '2-room', '3Gen', 'Adjoined flat', 'Apartment', 'DBSS', 
                'Improved', 'Improved-Maisonette', 'Maisonette', 'Model A', 
                'Model A-Maisonette', 'Model A2', 'Multi Generation', 
                'New Generation', 'Premium Apartment', 'Premium Apartment Loft', 
                'Premium Maisonette', 'Simplified', 'Standard', 'Terrace', 
                'Type S1', 'Type S2'
            ]
            model_index = model_options.index(default_model) if default_model in model_options else 8  # Default to Model A
            
            flat_model = st.selectbox(
                "🏗️ Flat Model",
                options=model_options,
                index=model_index,
                help="Select the flat model/design"
            )
            
            # Submit button
            submitted = st.form_submit_button("🎯 Predict Price", type="primary", use_container_width=True)
            
            # Clear example selection after form submission
            if submitted:
                st.session_state.example_selected = None
        
        # Quick examples moved after form
    
    with col2:
        st.subheader("📊 Prediction Results")
        
        # Make prediction when form is submitted
        if submitted:
            user_input = {
                'town': town,
                'flat_type': flat_type,
                'storey_range': storey_range,
                'floor_area_sqm': floor_area,
                'flat_model': flat_model,
                'remaining_lease': remaining_lease
            }
            
            try:
                predicted_price, warnings = predict_price(model_data, user_input)
                
                # Display warnings if any
                for warning in warnings:
                    st.warning(f"⚠️ {warning} - Using average value for prediction.")
                
                # Main prediction result
                st.success("🎯 **Prediction Complete!**")
                
                # Price display
                col_price1, col_price2, col_price3 = st.columns(3)
                
                with col_price2:
                    st.metric(
                        label="💰 Predicted Price",
                        value=f"${predicted_price:,.0f}",
                        help="Estimated resale price based on your inputs"
                    )
                
                # Price range
                lower_bound = predicted_price * 0.9
                upper_bound = predicted_price * 1.1
                
                st.info(f"📈 **Estimated Range:** ${lower_bound:,.0f} - ${upper_bound:,.0f}")
                
                # Confidence gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = predicted_price/1000,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Price (K SGD)"},
                    gauge = {
                        'axis': {'range': [None, 1000]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 300], 'color': "lightgray"},
                            {'range': [300, 600], 'color': "gray"},
                            {'range': [600, 1000], 'color': "darkgray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': predicted_price/1000
                        }
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
                
                # Input summary
                st.subheader("📋 Input Summary")
                summary_data = [
                    ["🏘️ Town", user_input['town']],
                    ["🏠 Flat Type", user_input['flat_type']],
                    ["🏢 Storey Range", user_input['storey_range']],
                    ["📐 Floor Area", f"{user_input['floor_area_sqm']:.1f} sqm"],
                    ["🏗️ Flat Model", user_input['flat_model']],
                    ["📅 Remaining Lease", f"{user_input['remaining_lease']:.1f} years"]
                ]
                
                summary_df = pd.DataFrame(summary_data, columns=["Feature", "Value"])
                st.table(summary_df)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
        
        elif submitted:
            st.warning("⚠️ Please fill in all the required fields.")
        
        else:
            # Welcome message
            st.info("👆 **Enter your flat details and click 'Predict Price' to get started!**")
            
            # Show feature importance chart
            if 'feature_importance' in model_data:
                st.subheader("📊 Feature Importance")
                
                importance_data = model_data['feature_importance']
                features = list(importance_data.keys())
                importances = list(importance_data.values())
                
                fig = px.bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    title="What factors matter most for pricing?",
                    labels={'x': 'Importance', 'y': 'Features'}
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p>⚠️ <strong>Disclaimer:</strong> This prediction is based on historical data and should be used as a reference only. 
    Actual market prices may vary due to market conditions, specific flat characteristics, and other factors.</p>
    <p>📊 Model trained on HDB resale data from 2020-2025 | 🤖 Powered by Random Forest ML</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()