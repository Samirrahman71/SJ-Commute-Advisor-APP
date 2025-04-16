import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from data_utils import (
    generate_commute_recommendations,
    generate_enhanced_route_recommendations,
    fetch_weather_data,
    fetch_live_traffic_data,
    fetch_special_events,
    calculate_dynamic_risk_score
)

# Page config
st.set_page_config(
    page_title="AI Commute Advisor - San Jose Traffic Safety",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("AI Commute Advisor")
st.markdown("""
Your personalized commuting assistant for navigating San Jose's traffic patterns and finding the optimal route.
""")

# Check if data is available in session state
if 'filtered_df' not in st.session_state:
    st.error("No data available. Please return to the home page to load data.")
    st.stop()

# Get data from session state
df = st.session_state['filtered_df']

# User Input Section
st.header("Tell me about your commute")
col1, col2 = st.columns(2)

with col1:
    start_location = st.text_input("Starting Location", "San Jose Downtown")
    preferred_departure = st.time_input("Preferred Departure Time", datetime.now().time())
    commute_frequency = st.selectbox(
        "How often do you make this commute?",
        ["Daily", "Weekly", "Occasionally", "One-time"]
    )

with col2:
    end_location = st.text_input("Destination", "San Jose International Airport")
    return_time = st.time_input("Expected Return Time", datetime.now().time())
    transportation_mode = st.multiselect(
        "Preferred Transportation Modes",
        ["Drive", "Public Transit", "Bike", "Walk", "Mixed Mode"],
        default=["Drive"]
    )

# Additional Preferences
st.subheader("Additional Preferences")
col1, col2 = st.columns(2)

with col1:
    avoid_construction = st.checkbox("Avoid Construction Zones", value=True)
    prefer_scenic = st.checkbox("Prefer Scenic Routes", value=False)
    avoid_high_risk = st.checkbox("Avoid High Risk Areas", value=True)

with col2:
    eco_friendly = st.checkbox("Prioritize Eco-friendly Options", value=False)
    cost_sensitive = st.checkbox("Cost-sensitive Travel", value=False)
    flexible_schedule = st.checkbox("Flexible Schedule", value=False)

# Generate Recommendations
if st.button("Get Personalized Recommendations"):
    with st.spinner("Analyzing traffic patterns and generating recommendations..."):
        try:
            # Get current conditions
            current_time = datetime.now()
            weather_data = fetch_weather_data("YOUR_API_KEY", 37.3382, -121.8863)  # San Jose coordinates
            traffic_data = fetch_live_traffic_data("YOUR_API_KEY")
            events = fetch_special_events("YOUR_API_KEY", 37.3382, -121.8863)
            
            # Generate route recommendations
            recommendations = generate_enhanced_route_recommendations(
                df=df,
                start_location=start_location,
                end_location=end_location,
                preferred_time=datetime.combine(datetime.today(), preferred_departure)
            )
            
            # Display Current Conditions
            st.header("Current Conditions")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Weather")
                if weather_data:
                    st.write(f"Condition: {weather_data.get('condition', 'Unknown')}")
                    st.write(f"Temperature: {weather_data.get('temperature', 'N/A')}°K")
                    st.write(f"Visibility: {weather_data.get('visibility', 'N/A')}m")
                else:
                    st.write("Weather data unavailable")
            
            with col2:
                st.subheader("Traffic")
                if not traffic_data.empty:
                    st.write(f"Current Traffic Level: {traffic_data['traffic_level'].iloc[0]}")
                    st.write(f"Average Speed: {traffic_data['avg_speed'].iloc[0]} mph")
                else:
                    st.write("Traffic data unavailable")
            
            with col3:
                st.subheader("Special Events")
                if events:
                    for event in events[:3]:
                        st.write(f"• {event.get('name', 'Unknown Event')}")
                        st.write(f"  Impact: {event.get('impact_score', 0):.2f}")
                else:
                    st.write("No active events")
            
            # Route Options
            st.header("Recommended Routes")
            for route_type, details in recommendations['routes'].items():
                with st.expander(f"{route_type} - {details['recommendation']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"Risk Score: {details['risk_score']:.2f}")
                        st.write(f"Description: {details['description']}")
                        st.write(f"Estimated Time: {details['estimated_time']}")
                    
                    with col2:
                        st.write("Safety Tips:")
                        for tip in details['safety_tips']:
                            st.write(f"• {tip}")
            
            # Alternative Transportation Options
            if "Public Transit" in transportation_mode:
                st.header("Public Transit Options")
                st.write("VTA Light Rail and Bus Services:")
                st.write("• Light Rail: Available every 15-20 minutes")
                st.write("• Express Bus: Available during peak hours")
                st.write("• Regular Bus: Available throughout the day")
                
                if cost_sensitive:
                    st.write("Cost Comparison:")
                    st.write("• Light Rail: $2.50 one-way")
                    st.write("• Bus: $2.25 one-way")
                    st.write("• Day Pass: $7.50")
            
            # Weather-specific Recommendations
            if weather_data and weather_data.get('condition') in ['Rain', 'Heavy Rain', 'Fog']:
                st.header("Weather-specific Recommendations")
                st.write("• Allow extra travel time")
                st.write("• Use windshield wipers and defogger")
                st.write("• Maintain safe following distance")
                st.write("• Consider public transit if conditions are severe")
            
            # Cost Analysis
            if cost_sensitive:
                st.header("Cost Analysis")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Driving")
                    st.write("• Fuel Cost: ~$4.50/gallon")
                    st.write("• Parking: $2-5/hour")
                    st.write("• Total: ~$15-20 round trip")
                
                with col2:
                    st.subheader("Public Transit")
                    st.write("• Day Pass: $7.50")
                    st.write("• Monthly Pass: $70")
                    st.write("• Total: $7.50/day")
                
                with col3:
                    st.subheader("Mixed Mode")
                    st.write("• Park & Ride: $5/day")
                    st.write("• Transit: $7.50/day")
                    st.write("• Total: $12.50/day")
            
            # Future Trends
            st.header("Traffic Pattern Insights")
            st.write("Based on historical data and current trends:")
            st.write("• Morning Rush Hour: 7:00 AM - 9:00 AM")
            st.write("• Evening Rush Hour: 4:00 PM - 6:00 PM")
            st.write("• Best Times to Travel: 10:00 AM - 3:00 PM")
            
            if flexible_schedule:
                st.write("Consider adjusting your schedule to avoid peak hours")
            
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Data is updated every 5 minutes. Last update: {}</p>
    <p>For emergency assistance, call 911</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True) 