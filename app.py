import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Custom page configuration
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        color: #333333;
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #1f4e79;
        color: white;
        height: 100vh; /* Full height */
        overflow-y: auto; /* Enable vertical scrolling */
    }
    h1 {
        color: #1f4e79;
    }
    .stButton button {
        background-color: #1f4e79;
        color: white;
        border-radius: 5px;
    }
    .stSlider .stSliderLabel, .stSelectbox label, .stNumberInput label {
        color: #1f4e79;
    }
    .stSidebar .stSidebarContent {
        padding: 1rem;
    }
    .st-expanderHeader {
        background-color: #1f4e79;
        color: white;
        font-weight: bold;
    }
    .st-expanderContent {
        background-color: #e3f2fd;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for configuration options
st.sidebar.image("https://freepngimg.com/download/laptop/162431-laptop-vector-notebook-png-free-photo.png", caption="Select Your Preferences", use_column_width=True)
st.sidebar.title("Additional Features")

# Sidebar inputs
touchscreen = st.sidebar.radio('Touchscreen', ['No', 'Yes'], help="Does the laptop have a touchscreen?")
ips = st.sidebar.radio('IPS Display', ['No', 'Yes'], help="Does the laptop have an IPS display?")

# Main screen inputs
st.title("🎯 Laptop Price Predictor")
st.write("#### Customize the specifications of your laptop to estimate its price.")

# Create an expander for each category to keep the UI clean
with st.expander("Laptop Specifications", expanded=True):
    # Inputs
    company = st.selectbox('Brand', df['Company'].unique(), help="Select the laptop brand.")
    type = st.selectbox('Type', df['TypeName'].unique(), help="Choose the type of laptop.")
    ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64], help="Select the RAM size.")
    weight = st.number_input('Weight (in kg)', help="Enter the laptop weight.")
    screen_size = st.slider('Screen Size (in inches)', 10.0, 18.0, 13.0, help="Adjust the screen size.")
    resolution = st.selectbox('Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
                                             '2880x1800', '2560x1600', '2560x1440', '2304x1440'], help="Choose the resolution.")
    cpu = st.selectbox('CPU', df['Cpu brand'].unique(), help="Choose the CPU brand.")
    gpu = st.selectbox('GPU', df['Gpu Brand'].unique(), help="Select the GPU brand.")

with st.expander("Storage & OS", expanded=True):
    # Inputs
    hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048], help="Choose the HDD size.")
    ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024], help="Choose the SSD size.")
    os = st.selectbox('Operating System', df['os'].unique(), help="Select the operating system.")

# Predict button
if st.button('💡 Predict Price'):
    # Processing
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Show progress
    st.progress(50)
    
    # Predicting
    predicted_price = np.exp(pipe.predict(query)[0])
    
    # Display result
    st.success(f"💻 The estimated price for this configuration is: **₹ {int(predicted_price):,}**")
