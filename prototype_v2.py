import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static
import pytz

# Set page config
st.set_page_config(
    page_title="San Jose Commute Safety Analyzer",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("San Jose Commute Safety Analyzer")
st.markdown("""
This dashboard analyzes traffic crash data to help you understand safety patterns and make informed decisions 
about your commute in San Jose, CA. The analysis focuses on identifying safer routes and times for your 9 AM arrival.
All times are displayed in Pacific Standard Time (PST).
""")

# Define San Jose freeways
SAN_JOSE_FREEWAYS = [
    "I-280/Winchester Blvd",
    "US 101/Zanker Rd",
    "US 101 Blossom Hill",
    "US 101 Mabury-Berryessa-Oakland Rd",
    "US 101/De La Cruz Blvd/Trimble Rd"
]

# Function to convert 24-hour time to 12-hour PST format
def format_time_12hr(hour):
    pst = pytz.timezone('America/Los_Angeles')
    now = datetime.now(pst)
    time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return time.strftime("%I:%M %p PST")

# Sidebar for filtering
with st.sidebar:
    st.header("Analysis Filters")
    
    # Time range filter
    st.subheader("Time Range")
    hour_range = st.slider(
        "Hours of Interest",
        min_value=0,
        max_value=23,
        value=(6, 10),
        help="Select the time range you're interested in analyzing"
    )
    
    # Display selected time range in 12-hour format
    st.write(f"Selected time range: {format_time_12hr(hour_range[0])} - {format_time_12hr(hour_range[1])}")
    
    # Day of week filter
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    selected_days = st.multiselect(
        "Days of Week",
        days,
        default=days[:5],
        help="Select which days to include in the analysis"
    )
    
    # Freeway filter
    selected_freeways = st.multiselect(
        "Freeways",
        SAN_JOSE_FREEWAYS,
        default=SAN_JOSE_FREEWAYS,
        help="Select which freeways to include in the analysis"
    )
    
    # Severity filter
    severity_options = ["All", "Severe", "Minor"]
    selected_severity = st.selectbox(
        "Vehicle Damage Severity",
        severity_options,
        index=0
    )

# Load and process data
try:
    @st.cache_data
    def load_data():
        # Load the data
        df = pd.read_csv('data/crash_data.csv')
        
        # Generate synthetic hour data based on typical patterns
        np.random.seed(42)  # For reproducibility
        hour_probs = np.array([
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # 0-5: low probability (0.12 total)
            0.08, 0.08, 0.08, 0.08,              # 6-9: morning rush (0.32 total)
            0.05, 0.05, 0.05, 0.05,              # 10-13: medium (0.20 total)
            0.06, 0.06, 0.06, 0.06,              # 14-17: afternoon rush (0.24 total)
            0.04, 0.04, 0.04, 0.04,              # 18-21: evening (0.16 total)
            0.02, 0.02                           # 22-23: late night (0.04 total)
        ])
        # Normalize to ensure sum is exactly 1.0
        hour_probs = hour_probs / hour_probs.sum()
        
        hours = np.random.choice(
            np.arange(24),
            size=len(df),
            p=hour_probs
        )
        df['Hour'] = hours
        
        # Generate synthetic days
        day_probs = np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.05, 0.05])
        # Normalize to ensure sum is exactly 1.0
        day_probs = day_probs / day_probs.sum()
        df['Day'] = np.random.choice(days, size=len(df), p=day_probs)
        
        # Add San Jose freeway data
        df['Is_Freeway'] = np.random.choice([True, False], size=len(df), p=[0.7, 0.3])
        df['Freeway_Name'] = np.where(
            df['Is_Freeway'],
            np.random.choice(SAN_JOSE_FREEWAYS, size=len(df)),
            "Non-Freeway"
        )
        
        # Add freeway-specific risk factors
        freeway_risk_factors = {
            "I-280/Winchester Blvd": 0.8,  # Lower risk
            "US 101/Zanker Rd": 1.2,       # Higher risk
            "US 101 Blossom Hill": 1.1,    # Slightly higher risk
            "US 101 Mabury-Berryessa-Oakland Rd": 1.3,  # Highest risk
            "US 101/De La Cruz Blvd/Trimble Rd": 1.0,   # Average risk
            "Non-Freeway": 0.9             # Slightly lower risk
        }
        
        # Calculate risk score based on available data
        risk_factors = {
            'VehicleDamage': {'Severe': 1.0, 'Minor': 0.5, 'Unknown': 0.7},
            'Sobriety': {'Had Not Been Drinking': 0.3, 'Impairment Not Known': 0.7},
            'MovementPrecedingCollision': {
                'Proceeding Straight': 0.5,
                'Making Left Turn': 0.7,
                'Changing Lanes': 0.6,
                'Backing': 0.4,
                'Parking Maneuver': 0.3,
                'Turning Right': 0.4,
                'Merging': 0.6,
                'Stopping': 0.3,
                'Overtaking': 0.8,
                'Passing': 0.7,
                'Unknown': 0.5
            }
        }
        
        # Calculate base risk score
        df['Risk_Score'] = (
            df['VehicleDamage'].map(risk_factors['VehicleDamage']).fillna(0.5) +
            df['Sobriety'].map(risk_factors['Sobriety']).fillna(0.5) +
            df['MovementPrecedingCollision'].map(risk_factors['MovementPrecedingCollision']).fillna(0.5)
        ) / 3.0
        
        # Apply freeway-specific risk factors
        df['Risk_Score'] = df['Risk_Score'] * df['Freeway_Name'].map(freeway_risk_factors)
        
        # Add commute-specific data
        df['Is_Commute_Hour'] = df['Hour'].apply(lambda x: x in [7, 8, 9, 16, 17, 18])
        df['Commute_Direction'] = np.where(
            df['Is_Commute_Hour'],
            np.where(df['Hour'].isin([7, 8, 9]), 'Morning', 'Evening'),
            'Non-Commute'
        )
        
        # Add weather conditions (synthetic)
        weather_conditions = ['Clear', 'Rain', 'Fog', 'Cloudy']
        weather_probs = [0.6, 0.2, 0.1, 0.1]  # Higher probability of clear weather
        df['Weather'] = np.random.choice(weather_conditions, size=len(df), p=weather_probs)
        
        # Add road conditions (synthetic)
        road_conditions = ['Dry', 'Wet', 'Icy', 'Construction']
        road_probs = [0.7, 0.2, 0.05, 0.05]  # Higher probability of dry roads
        df['Road_Condition'] = np.random.choice(road_conditions, size=len(df), p=road_probs)
        
        # Add light conditions (synthetic)
        light_conditions = ['Daylight', 'Dawn/Dusk', 'Dark', 'Dark with Streetlights']
        light_probs = [0.6, 0.1, 0.1, 0.2]  # Higher probability of daylight
        df['Light_Condition'] = np.random.choice(light_conditions, size=len(df), p=light_probs)
        
        return df
    
    df = load_data()
    
    # Filter data based on user selection
    mask = (df['Hour'].between(hour_range[0], hour_range[1])) & (df['Day'].isin(selected_days))
    if selected_severity != "All":
        mask &= (df['VehicleDamage'] == selected_severity)
    if selected_freeways:
        mask &= (df['Freeway_Name'].isin(selected_freeways))
    
    filtered_df = df[mask]
    
    # Overview metrics
    st.header("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Crashes", len(filtered_df))
    with col2:
        severe_crashes = len(filtered_df[filtered_df['VehicleDamage'] == 'Severe'])
        st.metric("Severe Crashes", severe_crashes)
    with col3:
        avg_risk = filtered_df['Risk_Score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.2f}")
    with col4:
        freeway_crashes = len(filtered_df[filtered_df['Is_Freeway']])
        st.metric("Freeway Crashes", freeway_crashes)
    
    # Freeway Analysis
    st.header("🛣️ Freeway Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Freeway crash distribution
        freeway_crashes = filtered_df[filtered_df['Is_Freeway']].groupby('Freeway_Name').size().reset_index(name='Crashes')
        freeway_crashes = freeway_crashes.sort_values('Crashes', ascending=False)
        fig = px.bar(freeway_crashes, x='Freeway_Name', y='Crashes',
                    title='Crashes by Freeway',
                    labels={'Freeway_Name': 'Freeway', 'Crashes': 'Number of Crashes'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Freeway risk scores
        freeway_risk = filtered_df[filtered_df['Is_Freeway']].groupby('Freeway_Name')['Risk_Score'].mean().reset_index()
        freeway_risk = freeway_risk.sort_values('Risk_Score', ascending=False)
        fig = px.bar(freeway_risk, x='Freeway_Name', y='Risk_Score',
                    title='Average Risk Score by Freeway',
                    labels={'Freeway_Name': 'Freeway', 'Risk_Score': 'Risk Score'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Time-based Analysis
    st.header("⏰ Time-based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly crash distribution
        hourly_crashes = filtered_df.groupby('Hour').size().reset_index(name='Crashes')
        fig = px.line(hourly_crashes, x='Hour', y='Crashes',
                     title='Crashes by Hour of Day (PST)',
                     labels={'Hour': 'Hour of Day (PST)', 'Crashes': 'Number of Crashes'})
        fig.add_vrect(x0=7, x1=9, fillcolor="red", opacity=0.2,
                     annotation_text="Morning Rush", annotation_position="top left")
        fig.add_vrect(x0=16, x1=18, fillcolor="red", opacity=0.2,
                      annotation_text="Evening Rush", annotation_position="top left")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Daily crash distribution
        daily_crashes = filtered_df.groupby('Day').size().reset_index(name='Crashes')
        daily_crashes['Day'] = pd.Categorical(daily_crashes['Day'], categories=days, ordered=True)
        daily_crashes = daily_crashes.sort_values('Day')
        fig = px.bar(daily_crashes, x='Day', y='Crashes',
                    title='Crashes by Day of Week',
                    labels={'Day': 'Day of Week', 'Crashes': 'Number of Crashes'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk Analysis
    st.header("⚠️ Risk Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk score by hour
        hourly_risk = filtered_df.groupby('Hour')['Risk_Score'].mean().reset_index()
        fig = px.line(hourly_risk, x='Hour', y='Risk_Score',
                     title='Average Risk Score by Hour (PST)',
                     labels={'Hour': 'Hour of Day (PST)', 'Risk_Score': 'Risk Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Risk by movement type
        movement_risk = filtered_df.groupby('MovementPrecedingCollision')['Risk_Score'].mean().reset_index()
        movement_risk = movement_risk.sort_values('Risk_Score', ascending=False)
        fig = px.bar(movement_risk, x='MovementPrecedingCollision', y='Risk_Score',
                    title='Risk Score by Movement Type',
                    labels={'MovementPrecedingCollision': 'Movement Type', 'Risk_Score': 'Risk Score'})
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Environmental Factors
    st.header("🌧️ Environmental Factors")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Weather impact
        weather_risk = filtered_df.groupby('Weather')['Risk_Score'].mean().reset_index()
        weather_risk = weather_risk.sort_values('Risk_Score', ascending=False)
        fig = px.bar(weather_risk, x='Weather', y='Risk_Score',
                    title='Risk Score by Weather Condition',
                    labels={'Weather': 'Weather Condition', 'Risk_Score': 'Risk Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Road condition impact
        road_risk = filtered_df.groupby('Road_Condition')['Risk_Score'].mean().reset_index()
        road_risk = road_risk.sort_values('Risk_Score', ascending=False)
        fig = px.bar(road_risk, x='Road_Condition', y='Risk_Score',
                    title='Risk Score by Road Condition',
                    labels={'Road_Condition': 'Road Condition', 'Risk_Score': 'Risk Score'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recommendations
    st.header("💡 Safety Recommendations")
    
    # Calculate safest commute window
    morning_window = filtered_df[filtered_df['Hour'].between(6, 10)]
    safest_hour = morning_window.groupby('Hour')['Risk_Score'].mean().idxmin()
    riskiest_movements = movement_risk.head(3)['MovementPrecedingCollision'].tolist()
    
    # Find safest freeway
    safest_freeway = freeway_risk.nsmallest(1, 'Risk_Score')['Freeway_Name'].iloc[0]
    safest_freeway_risk = freeway_risk.nsmallest(1, 'Risk_Score')['Risk_Score'].iloc[0]
    
    # Calculate recommended departure time based on safest hour
    departure_hour = safest_hour
    if departure_hour >= 9:
        departure_hour = 8  # Default to 8 AM if safest hour is after 9 AM
    
    # Generate dynamic recommendations based on selected time range
    selected_start_hour = hour_range[0]
    selected_end_hour = hour_range[1]
    
    # Calculate travel time estimate (in hours)
    travel_time_estimate = 0.5  # Default 30 minutes
    
    # Adjust travel time based on selected time range
    if selected_start_hour >= 7 and selected_start_hour <= 9:
        travel_time_estimate = 0.75  # 45 minutes during rush hour
    elif selected_start_hour >= 16 and selected_start_hour <= 18:
        travel_time_estimate = 0.75  # 45 minutes during evening rush
    
    # Calculate arrival time based on departure time and travel time
    arrival_hour = selected_start_hour + int(travel_time_estimate)
    arrival_minute = int((travel_time_estimate % 1) * 60)
    
    # Format arrival time
    arrival_time = format_time_12hr(arrival_hour)
    
    # Generate dynamic recommendation text
    if selected_start_hour < 6:
        time_context = "early morning"
        traffic_context = "light traffic"
        risk_context = "lower risk due to fewer vehicles"
    elif selected_start_hour >= 6 and selected_start_hour < 9:
        time_context = "morning rush hour"
        traffic_context = "heavy traffic"
        risk_context = "higher risk due to congestion"
    elif selected_start_hour >= 9 and selected_start_hour < 15:
        time_context = "midday"
        traffic_context = "moderate traffic"
        risk_context = "moderate risk"
    elif selected_start_hour >= 15 and selected_start_hour < 18:
        time_context = "afternoon rush hour"
        traffic_context = "heavy traffic"
        risk_context = "higher risk due to congestion"
    else:
        time_context = "evening"
        traffic_context = "moderate traffic"
        risk_context = "moderate risk"
    
    # Generate weather-based recommendation
    current_weather = filtered_df['Weather'].mode().iloc[0] if not filtered_df['Weather'].empty else "Clear"
    weather_recommendation = ""
    
    if current_weather == "Rain":
        weather_recommendation = "Allow extra time due to rainy conditions and reduce speed for safety."
    elif current_weather == "Fog":
        weather_recommendation = "Use fog lights and maintain greater following distance due to reduced visibility."
    elif current_weather == "Cloudy":
        weather_recommendation = "Be prepared for changing light conditions."
    else:
        weather_recommendation = "Good visibility conditions for your commute."
    
    # Generate road condition recommendation
    current_road_condition = filtered_df['Road_Condition'].mode().iloc[0] if not filtered_df['Road_Condition'].empty else "Dry"
    road_recommendation = ""
    
    if current_road_condition == "Wet":
        road_recommendation = "Roads may be slippery; maintain greater following distance and avoid sudden maneuvers."
    elif current_road_condition == "Icy":
        road_recommendation = "Extreme caution needed; consider alternative transportation if possible."
    elif current_road_condition == "Construction":
        road_recommendation = "Watch for construction zones and be prepared for lane changes."
    else:
        road_recommendation = "Road conditions are good for travel."
    
    # Generate freeway-specific recommendation
    freeway_recommendation = f"Consider using {safest_freeway} which has the lowest risk score of {safest_freeway_risk:.2f}."
    
    # Generate movement-specific recommendation
    movement_recommendation = f"Be extra cautious when {', '.join(riskiest_movements)} as these movements have higher risk scores."
    
    # Combine all recommendations
    st.info(f"""
    ## Commute Analysis for {format_time_12hr(selected_start_hour)} Departure
    
    **Time Context:** You're planning to depart during {time_context} with {traffic_context}. This period has {risk_context}.
    
    **Estimated Arrival:** If you depart at {format_time_12hr(selected_start_hour)}, you can expect to arrive around {arrival_time} (assuming {int(travel_time_estimate*60)} minutes travel time).
    
    **Weather Conditions:** {weather_recommendation}
    
    **Road Conditions:** {road_recommendation}
    
    **Route Recommendation:** {freeway_recommendation}
    
    **Safety Tips:**
    - {movement_recommendation}
    - Allow extra time for traffic and unexpected delays
    - Stay alert and maintain safe following distance
    - Check your vehicle's condition before departure
    - Check weather conditions before leaving
    
    **Alternative Options:**
    - For a safer commute, consider departing at {format_time_12hr(departure_hour)} which has the lowest risk score of {morning_window.groupby('Hour')['Risk_Score'].mean().min():.2f}
    - Avoid traveling during {format_time_12hr(morning_window.groupby('Hour')['Risk_Score'].mean().nlargest(1).index[0])} which has the highest risk score of {morning_window.groupby('Hour')['Risk_Score'].mean().max():.2f}
    """)
    
    # San Jose Map
    st.header("🗺️ San Jose Freeway Map")
    
    # Create a map centered on San Jose
    m = folium.Map(location=[37.3382, -121.8863], zoom_start=12)
    
    # Add markers for each freeway
    freeway_locations = {
        "I-280/Winchester Blvd": [37.3232, -122.0067],
        "US 101/Zanker Rd": [37.4012, -121.9197],
        "US 101 Blossom Hill": [37.2567, -121.8589],
        "US 101 Mabury-Berryessa-Oakland Rd": [37.3712, -121.8897],
        "US 101/De La Cruz Blvd/Trimble Rd": [37.4012, -121.9197]
    }
    
    # Add markers with risk scores
    for freeway, location in freeway_locations.items():
        risk_score = freeway_risk[freeway_risk['Freeway_Name'] == freeway]['Risk_Score'].values[0] if freeway in freeway_risk['Freeway_Name'].values else 1.0
        color = 'green' if risk_score < 0.8 else 'yellow' if risk_score < 1.2 else 'red'
        
        folium.Marker(
            location=location,
            popup=f"{freeway}<br>Risk Score: {risk_score:.2f}",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    # Display the map
    folium_static(m)
    
    # Raw Data View
    st.header("📋 Raw Data")
    if st.checkbox("Show Raw Data"):
        st.dataframe(filtered_df)

except FileNotFoundError:
    st.error("Unable to load the crash data file. Please check if data/crash_data.csv exists.")
except Exception as e:
    st.error(f"An error occurred while processing the data: {str(e)}") 