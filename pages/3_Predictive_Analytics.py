import streamlit as st
from datetime import datetime
from data_utils import generate_enhanced_route_recommendations

# Page config
st.set_page_config(
    page_title="Predictive Analytics - San Jose Traffic Safety",
    page_icon="🔮",
    layout="wide"
)

# Title
st.title("Predictive Analytics")
st.markdown("""
Get real-time risk predictions and route recommendations based on current conditions.
""")

# Check if data is available in session state
if 'filtered_df' not in st.session_state:
    st.error("No data available. Please return to the home page to load data.")
    st.stop()

# Get data from session state
df = st.session_state['filtered_df']

# Input fields for route planning
st.header("Route Planning")
col1, col2 = st.columns(2)
with col1:
    start_location = st.text_input("Starting Location", "San Jose Downtown")
with col2:
    end_location = st.text_input("Destination", "San Jose International Airport")

# Time selection
preferred_time = st.time_input("Preferred Departure Time", datetime.now().time())

if st.button("Get Route Recommendations"):
    try:
        # Generate enhanced route recommendations
        recommendations = generate_enhanced_route_recommendations(
            df=df,
            start_location=start_location,
            end_location=end_location,
            preferred_time=datetime.combine(datetime.today(), preferred_time)
        )
        
        # Display current conditions
        st.subheader("Current Conditions")
        conditions = recommendations['current_conditions']
        st.write(f"Weather: {conditions['weather']}")
        st.write(f"Road Condition: {conditions['road_condition']}")
        st.write(f"Time: {conditions['time']}")
        
        # Display route recommendations
        st.subheader("Route Options")
        for route_type, details in recommendations['routes'].items():
            with st.expander(f"{route_type} - {details['recommendation']}"):
                st.write(f"Risk Score: {details['risk_score']:.2f}")
                st.write(f"Description: {details['description']}")
                st.write(f"Estimated Time: {details['estimated_time']}")
                
                st.write("Safety Tips:")
                for tip in details['safety_tips']:
                    st.write(f"- {tip}")
                    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

# Add model training section (for administrators)
st.header("Model Training")
st.markdown("""
This section allows administrators to train and update the risk prediction model.
""")

if st.checkbox("Show Advanced Options"):
    if st.button("Train Risk Prediction Model"):
        with st.spinner("Training model..."):
            try:
                from data_utils import train_risk_predictor
                train_risk_predictor(df)
                st.success("Model trained successfully!")
            except Exception as e:
                st.error(f"Error training model: {str(e)}") 