import streamlit as st
import pickle
import numpy as np
import pandas as pd

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Laptop Price Predictor")

# Input widgets
company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM (in GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the Laptop (kg)', min_value=0.1, max_value=10.0, value=2.0)
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size (inches)', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', 
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (in GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (in GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

def calculate_ppi(resolution, screen_size):
    try:
        x_res, y_res = map(int, resolution.split('x'))
        return ((x_res**2) + (y_res**2))**0.5 / screen_size
    except:
        return 0  # default value if calculation fails

if st.button('Predict Price'):
    # Process inputs
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips_value = 1 if ips == 'Yes' else 0  # Changed variable name to avoid conflict
    ppi = calculate_ppi(resolution, screen_size)
    
    # Create a DataFrame with the correct feature names (matching training data exactly)
    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips_value],  # Changed to match expected column name
        'ppi': [ppi],       # Changed to lowercase to match expected column name
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })
    
    # Make prediction
    try:
        pred = np.exp(pipe.predict(query)[0])
        formatted_price = "₹{:,.2f}".format(pred)
        st.success(f"The predicted price is {formatted_price}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
        st.write("Query DataFrame columns:", query.columns.tolist())  # Debug output