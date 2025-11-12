import streamlit as st
import joblib
import pandas as pd
import numpy as np 
import warnings
warnings.filterwarnings('ignore') # Suppress warnings during prediction

# --- Configuration ---
MODEL_PATH = 'laptop_price_model.joblib' 
TITLE = '💻 Expert Laptop Price Predictor'

# --- Feature Definitions (CORRECTED to include all 12 model features) ---
FEATURE_DEFS = {
    'Company': {'type': 'selectbox', 'options': ['Dell', 'HP', 'Lenovo', 'Apple', 'Asus', 'Microsoft', 'Acer', 'MSI', 'Other Brand']},
    'TypeName': {'type': 'selectbox', 'options': ['Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook', 'Other Type']},
    'Ram': {'type': 'selectbox', 'options': [4, 8, 16, 32, 64], 'unit': 'GB'},
    'Weight': {'type': 'number_input', 'min_value': 0.5, 'max_value': 5.0, 'default': 1.8, 'step': 0.1, 'format': '%.2f', 'unit': 'kg'},
    'TouchScreen': {'type': 'selectbox', 'options': [0, 1], 'labels': ['No (0)', 'Yes (1)'], 'default': 0},
    'Ips': {'type': 'selectbox', 'options': [0, 1], 'labels': ['No (0)', 'Yes (1)'], 'default': 0},
    'Ppi': {'type': 'number_input', 'min_value': 50.0, 'max_value': 450.0, 'default': 141.2, 'step': 0.1, 'format': '%.2f', 'unit': 'PPI'},
    # NEW/CORRECTED FEATURES:
    'Cpu_brand': {'type': 'selectbox', 'options': ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'Other Intel Processor', 'AMD Processor']},
    'HDD': {'type': 'number_input', 'min_value': 0, 'max_value': 2000, 'default': 0, 'step': 128, 'format': '%d', 'unit': 'GB'},
    'SSD': {'type': 'number_input', 'min_value': 0, 'max_value': 2000, 'default': 256, 'step': 128, 'format': '%d', 'unit': 'GB'},
    'Gpu_brand': {'type': 'selectbox', 'options': ['Intel', 'Nvidia', 'AMD', 'Other GPU']},
    'Os': {'type': 'selectbox', 'options': ['Windows', 'Mac', 'Linux', 'No OS']}
}

# 1. Load the model safely and efficiently using st.cache_resource
@st.cache_resource
def load_model(path):
    """Loads the scikit-learn pipeline model using joblib and caches it."""
    try:
        pipeline = joblib.load(path)
        st.success(f"Model Pipeline loaded successfully from **{path}**.")
        return pipeline
    except FileNotFoundError:
        st.error(f"Error: Model file not found at '{path}'. Please ensure you save your pipeline as 'laptop_price_model.joblib'.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model. Error: {e}")
        return None

def main():
    # 2. Display a clear title for the application
    st.set_page_config(page_title=TITLE, layout="wide")
    st.title(TITLE)
    st.markdown("Enter the laptop specifications below to get an estimated price.")
    st.markdown("---")

    pipeline = load_model(MODEL_PATH)

    if pipeline is None:
        return

    st.subheader("🛠️ Configure Laptop Features (12 Required Inputs)")
    
    # 3. Create user-friendly input widgets
    # Using three columns to organize 12 features neatly
    col1, col2, col3 = st.columns(3)
    input_data = {}
    
    # Dynamically create widgets based on the feature definitions
    for i, (feature, definition) in enumerate(FEATURE_DEFS.items()):
        
        current_col = [col1, col2, col3][i % 3] # Distribute features across 3 columns
        
        with current_col:
            # Clean up labels for display
            label = feature.replace('Ppi', 'Screen PPI').replace('Ram', 'RAM').replace('Cpu_brand', 'CPU Brand').replace('Gpu_brand', 'GPU Brand')
            
            if definition['type'] == 'selectbox':
                
                if feature in ['TouchScreen', 'Ips']:
                    # Binary features (0/1) with user-friendly labels
                    user_label = st.selectbox(
                        f"**{label}**",
                        options=definition['labels'],
                        index=definition['default']
                    )
                    # Store the actual numeric value (0 or 1) for the model
                    input_data[feature] = definition['options'][definition['labels'].index(user_label)]
                else:
                    input_data[feature] = st.selectbox(
                        f"**{label}** ({definition.get('unit', '')})",
                        options=definition['options']
                    )

            elif definition['type'] == 'number_input':
                input_data[feature] = st.number_input(
                    f"**{label}** ({definition.get('unit', '')})",
                    min_value=definition['min_value'],
                    max_value=definition['max_value'],
                    value=definition['default'],
                    step=definition.get('step', 1),
                    format=definition['format']
                )

    st.markdown("---")
    
    # 4. Include a "Predict" button
    if st.button("PREDICT LAPTOP PRICE", type="primary"):
        
        # 5. Collect user inputs, create Pandas DataFrame, and use the pipeline
        try:
            # Create a DataFrame with ALL 12 required columns
            input_df = pd.DataFrame([input_data])
            
            # Predict the log-transformed price
            log_price_pred = pipeline.predict(input_df).flatten()[0]
            
            # Convert back to the actual price
            predicted_price = np.exp(log_price_pred)
            
            # 6. Display the final prediction
            st.subheader("✅ Prediction Result")
            
            st.metric(
                label="Estimated Price (in USD)", 
                value=f"${predicted_price:,.2f}"
            )

            st.info(f"The underlying model predicted a **log price** of **{log_price_pred:.4f}**.")
            
        except ValueError as ve:
            # Specific error handling for missing columns (though this should now be fixed)
            st.error(f"Prediction Error: The model is still missing columns or receiving them in an incompatible format. Please ensure all 12 inputs are configured correctly. Details: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

if __name__ == '__main__':
    main()