import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from data_utils import (
    generate_synthetic_data,
    process_traffic_data,
    calculate_risk_score,
    format_time_12hr,
    generate_enhanced_route_recommendations,
    fetch_weather_data,
    fetch_special_events
)

# Page config
st.set_page_config(
    page_title="San Jose Traffic Safety Advisor",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 San Jose Traffic Safety Advisor")
st.markdown("""
    Analyze traffic patterns and get personalized route recommendations based on real-time conditions.
""")

# Load and process data
@st.cache_data(ttl=300)
def load_data():
    """Load and process traffic data"""
    try:
        df = generate_synthetic_data(n_samples=1000)
        df = process_traffic_data(df)
        
        # Store current conditions in session state
        st.session_state.current_weather = fetch_weather_data()
        st.session_state.active_events = fetch_special_events()
        st.session_state.last_updated = datetime.now(pytz.timezone('America/Los_Angeles')).strftime("%I:%M %p PST")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load data
df = load_data()

# Check if data is available
if df.empty:
    st.error("No data available. Please try refreshing the page.")
    st.stop()

# Sidebar filters
with st.sidebar:
    st.header("Analysis Filters")
    
    # Time range filter
    st.subheader("Time Range")
    time_range = st.slider(
        "Select hours",
        min_value=0,
        max_value=23,
        value=(6, 20),  # Default to 6 AM - 8 PM
        format="%I %p"
    )
    
    # Day of week filter
    st.subheader("Day of Week")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_days = st.multiselect(
        "Select days",
        days,
        default=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    )

# Filter data based on selections
filtered_df = df[
    (df['Hour'].between(time_range[0], time_range[1])) &
    (df['Day'].isin(selected_days))
]

# Data Analysis Section
st.header("📊 Traffic Pattern Analysis")

# Create two columns for metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Risk Score",
        f"{filtered_df['Risk_Score'].mean():.2f}",
        help="Average risk score across all incidents"
    )

with col2:
    peak_hour = filtered_df.groupby('Hour')['Risk_Score'].mean().idxmax()
    st.metric(
        "Highest Risk Hour",
        format_time_12hr(peak_hour),
        help="Hour with the highest average risk score"
    )

with col3:
    safest_hour = filtered_df.groupby('Hour')['Risk_Score'].mean().idxmin()
    st.metric(
        "Safest Hour",
        format_time_12hr(safest_hour),
        help="Hour with the lowest average risk score"
    )

# Risk Score Distribution
st.subheader("Risk Score Distribution by Hour")
hourly_risk = filtered_df.groupby('Hour')['Risk_Score'].mean().reset_index()
fig = px.line(
    hourly_risk,
    x='Hour',
    y='Risk_Score',
    title='Average Risk Score Throughout the Day',
    labels={'Hour': 'Hour of Day', 'Risk_Score': 'Risk Score'}
)
fig.update_layout(xaxis_tickformat='%I %p')
st.plotly_chart(fig, use_container_width=True)

# Route Advisory Section
st.header("🗺️ Route Advisory")

# Input fields for route planning
col1, col2 = st.columns(2)

with col1:
    start_location = st.text_input("Starting Location", "Downtown San Jose")
    
with col2:
    end_location = st.text_input("Destination", "North San Jose")

# Time selection
departure_time = st.time_input(
    "Planned Departure Time",
    value=datetime.now().time(),
    help="Select your preferred departure time"
)

if st.button("Get Route Recommendations", type="primary"):
    try:
        # Convert time input to datetime
        current_date = datetime.now().date()
        departure_datetime = datetime.combine(current_date, departure_time)
        
        # Generate recommendations
        recommendations = generate_enhanced_route_recommendations(
            filtered_df,
            start_location,
            end_location,
            departure_datetime
        )
        
        if 'error' in recommendations:
            st.error(recommendations['error'])
        else:
            # Display current conditions
            st.subheader("Current Conditions")
            conditions = recommendations['current_conditions']
            cond1, cond2, cond3 = st.columns(3)
            
            with cond1:
                st.info(f"🌤️ Weather: {conditions['weather']}")
            with cond2:
                st.info(f"🛣️ Road: {conditions['road_condition']}")
            with cond3:
                st.info(f"🕒 Time: {conditions['time']}")
            
            # Display route options
            st.subheader("Recommended Routes")
            for route_type, route_info in recommendations['routes'].items():
                with st.expander(f"{route_type} Route - {route_info['recommendation']}"):
                    # Risk score indicator
                    risk_color = (
                        "🟢" if route_info['risk_score'] < 0.3
                        else "🟡" if route_info['risk_score'] < 0.6
                        else "🔴"
                    )
                    
                    st.markdown(f"""
                        **Risk Level:** {risk_color} ({route_info['risk_score']:.2f})  
                        **Estimated Time:** {route_info['estimated_time']}  
                        **Description:** {route_info['description']}
                    """)
                    
                    # Safety tips in a clean format
                    st.markdown("**Safety Tips:**")
                    for tip in route_info['safety_tips']:
                        st.markdown(f"• {tip}")
                    
                    if route_info['alternatives']:
                        st.markdown("**Alternative Options:**")
                        for alt in route_info['alternatives']:
                            st.markdown(f"• {alt}")
    
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")

# Footer with data freshness
st.markdown("---")
if 'last_updated' in st.session_state:
    st.caption(f"Data last updated: {st.session_state.last_updated}") 