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
        
        **Accuracy:** Based on recent market data
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
        else:
            default_town = 'TAMPINES'
            default_flat_type = '4 ROOM'
            default_storey = '07 TO 09'
            default_area = 95.0
            default_model = 'Model A'
            default_lease = 85.0
        
        # Create form for better UX
        with st.form("prediction_form"):
            # Town selection with popular options first
            popular_towns = ['TAMPINES', 'JURONG WEST', 'BEDOK', 'WOODLANDS', 'YISHUN', 'HOUGANG']
            other_towns = ['ANG MO KIO', 'BUKIT MERAH', 'BUKIT TIMAH', 'CLEMENTI', 'PUNGGOL', 'SENGKANG', 'TOA PAYOH']
            all_towns = popular_towns + ['---'] + other_towns
            
            # Get index for default town
            town_index = all_towns.index(default_town) if default_town in all_towns else 0
            
            town = st.selectbox(
                "🏘️ Town",
                options=all_towns,
                index=town_index,
                help="Select the town where the flat is located"
            )
            
            flat_type_options = ['3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', '2 ROOM', '1 ROOM']
            flat_type_index = flat_type_options.index(default_flat_type) if default_flat_type in flat_type_options else 1
            
            flat_type = st.selectbox(
                "🏠 Flat Type",
                options=flat_type_options,
                index=flat_type_index,
                help="Select the type of flat"
            )
            
            storey_options = ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27', '28 TO 30']
            storey_index = storey_options.index(default_storey) if default_storey in storey_options else 2
            
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
                    max_value=250.0,
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
            
            model_options = ['Model A', 'Improved', 'New Generation', 'Premium Apartment', 'Simplified', 'Standard', 'Model A2', 'Adjoined Flat', 'Terrace']
            model_index = model_options.index(default_model) if default_model in model_options else 0
            
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
        if submitted and town != '---':
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
        
        elif submitted and town == '---':
            st.warning("⚠️ Please select a valid town from the dropdown.")
        
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
