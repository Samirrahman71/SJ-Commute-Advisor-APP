import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import pytz
import requests
from typing import Dict, List, Optional, Tuple
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic traffic crash data for San Jose."""
    np.random.seed(42)  # For reproducibility
    
    # Generate dates for the last year
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(years=1)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Generate hours with realistic distribution
    hour_probs = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # 0-5: low probability (0.12 total)
        0.08, 0.08, 0.08, 0.08,              # 6-9: morning rush (0.32 total)
        0.05, 0.05, 0.05, 0.05,              # 10-13: medium (0.20 total)
        0.06, 0.06, 0.06, 0.06,              # 14-17: afternoon rush (0.24 total)
        0.04, 0.04, 0.04, 0.04,              # 18-21: evening (0.16 total)
        0.02, 0.02                           # 22-23: late night (0.04 total)
    ])
    hour_probs = hour_probs / hour_probs.sum()
    
    # Generate days with realistic distribution
    day_probs = np.array([0.18, 0.18, 0.18, 0.18, 0.18, 0.05, 0.05])  # Weekdays vs weekends
    day_probs = day_probs / day_probs.sum()
    
    # San Jose freeways and their approximate coordinates
    freeway_locations = {
        'US-101': {'lat_range': (37.3095, 37.4095), 'lon_range': (-121.9163, -121.8163)},
        'I-280': {'lat_range': (37.3082, 37.4082), 'lon_range': (-121.9251, -121.8251)},
        'I-680': {'lat_range': (37.2968, 37.3968), 'lon_range': (-121.8763, -121.7763)},
        'I-880': {'lat_range': (37.3182, 37.4182), 'lon_range': (-121.9051, -121.8051)},
        'SR-87': {'lat_range': (37.2895, 37.3895), 'lon_range': (-121.9063, -121.8063)},
        'SR-85': {'lat_range': (37.2768, 37.3768), 'lon_range': (-121.9351, -121.8351)},
        'SR-237': {'lat_range': (37.3882, 37.4882), 'lon_range': (-121.9651, -121.8651)}
    }
    
    # San Jose local area boundaries
    local_area = {
        'lat_range': (37.2500, 37.4500),  # San Jose latitude bounds
        'lon_range': (-122.0000, -121.7500)  # San Jose longitude bounds
    }
    
    # Generate locations (freeways and streets)
    is_freeway = np.random.choice([True, False], size=n_samples, p=[0.4, 0.6])
    locations = []
    latitudes = []
    longitudes = []
    freeway_names = []
    
    for is_fw in is_freeway:
        if is_fw:
            freeway = np.random.choice(list(freeway_locations.keys()))
            locations.append(freeway)
            freeway_names.append(freeway)
            lat_range = freeway_locations[freeway]['lat_range']
            lon_range = freeway_locations[freeway]['lon_range']
            latitudes.append(np.random.uniform(lat_range[0], lat_range[1]))
            longitudes.append(np.random.uniform(lon_range[0], lon_range[1]))
        else:
            street_name = f"Street_{np.random.randint(1, 100)}"
            locations.append(street_name)
            freeway_names.append('Local Road')
            latitudes.append(np.random.uniform(local_area['lat_range'][0], local_area['lat_range'][1]))
            longitudes.append(np.random.uniform(local_area['lon_range'][0], local_area['lon_range'][1]))
    
    # Generate risk factors with realistic probabilities
    risk_factors = [
        'Speeding',
        'Distracted Driving',
        'Weather Conditions',
        'Road Construction',
        'Traffic Congestion',
        'Poor Visibility',
        'Vehicle Malfunction'
    ]
    risk_probs = [0.3, 0.25, 0.15, 0.1, 0.1, 0.05, 0.05]
    risk_probs = np.array(risk_probs) / sum(risk_probs)
    
    # Generate environmental conditions with realistic probabilities
    weather_conditions = ['Clear', 'Rain', 'Fog', 'Wind']
    weather_probs = [0.7, 0.15, 0.1, 0.05]
    weather_probs = np.array(weather_probs) / sum(weather_probs)
    
    road_conditions = ['Dry', 'Wet', 'Icy', 'Construction']
    road_probs = [0.7, 0.15, 0.05, 0.1]
    road_probs = np.array(road_probs) / sum(road_probs)
    
    # Generate vehicle damage severity with realistic distribution
    vehicle_damage_probs = [0.6, 0.3, 0.1]  # Minor, Moderate, Severe
    vehicle_damage = np.random.choice(['Minor', 'Moderate', 'Severe'], size=n_samples, p=vehicle_damage_probs)
    
    # Generate movement preceding collision with realistic distribution
    movement_types = [
        'Proceeding Straight',
        'Making Left Turn',
        'Changing Lanes',
        'Backing',
        'Parking Maneuver',
        'Turning Right',
        'Merging',
        'Stopping',
        'Overtaking',
        'Passing',
        'Unknown'
    ]
    movement_probs = [0.3, 0.15, 0.1, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.03, 0.02]
    movement_probs = np.array(movement_probs) / sum(movement_probs)
    
    # Generate the DataFrame
    data = {
        'Date': dates,
        'Hour': np.random.choice(np.arange(24), size=n_samples, p=hour_probs),
        'Day': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                               size=n_samples, p=day_probs),
        'Location': locations,
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Is_Freeway': is_freeway,
        'Freeway_Name': freeway_names,
        'Risk_Factor': np.random.choice(risk_factors, size=n_samples, p=risk_probs),
        'Weather': np.random.choice(weather_conditions, size=n_samples, p=weather_probs),
        'Road_Condition': np.random.choice(road_conditions, size=n_samples, p=road_probs),
        'Severity': np.random.choice(['Minor', 'Moderate', 'Severe'], size=n_samples),
        'VehicleDamage': vehicle_damage,
        'MovementPrecedingCollision': np.random.choice(movement_types, size=n_samples, p=movement_probs),
        'Injuries': np.random.randint(0, 5, size=n_samples),
        'Vehicles_Involved': np.random.randint(1, 4, size=n_samples),
        'Crash_Number': range(1, n_samples + 1)
    }
    
    df = pd.DataFrame(data)
    
    # Add time-based features
    df['Is_Rush_Hour'] = df['Hour'].apply(lambda x: (x >= 7 and x <= 9) or (x >= 16 and x <= 18))
    df['Is_Night'] = df['Hour'].apply(lambda x: x < 6 or x > 18)
    
    # Add environmental risk factors
    df['Has_Adverse_Weather'] = df['Weather'].isin(['Rain', 'Fog'])
    df['Has_Adverse_Road'] = df['Road_Condition'].isin(['Wet', 'Icy', 'Construction'])
    
    # Calculate initial risk scores
    df['Risk_Score'] = calculate_risk_score(df)
    
    return df

def process_traffic_data(df):
    """Process traffic crash data and add derived features."""
    try:
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure Date column exists and is datetime
        if 'Date' not in df.columns:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            dates = [start_date + timedelta(days=x) for x in range(31)]
            df['Date'] = [np.random.choice(dates) for _ in range(len(df))]
        else:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure Hour is numeric
        if 'Hour' in df.columns:
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
        else:
            # Extract hour from time if available
            if 'Time' in df.columns:
                df['Hour'] = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce').dt.hour
            else:
                # Generate random hours if neither exists
                df['Hour'] = np.random.randint(0, 24, size=len(df))
        
        # Ensure Location column exists
        if 'Location' not in df.columns:
            df['Location'] = 'Unknown Location'
        
        # Add freeway information if not present
        if 'Is_Freeway' not in df.columns:
            df['Is_Freeway'] = df['Location'].str.contains('US-|I-|CA-', regex=True)
        
        if 'Freeway_Name' not in df.columns:
            df['Freeway_Name'] = df.apply(lambda x: extract_freeway_name(x['Location']) if x['Is_Freeway'] else 'Local Road', axis=1)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['Risk_Score', 'Injuries', 'Vehicles_Involved']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate risk score if not present
        if 'Risk_Score' not in df.columns:
            df['Risk_Score'] = df.apply(calculate_risk_score, axis=1)
        
        # Add time-based features
        df['Is_Rush_Hour'] = df['Hour'].apply(lambda x: (x >= 7 and x <= 9) or (x >= 16 and x <= 18))
        df['Is_Night'] = df['Hour'].apply(lambda x: x < 6 or x > 18)
        
        # Add environmental risk factors
        df['Has_Adverse_Weather'] = df['Weather'].isin(['Rain', 'Fog'])
        df['Has_Adverse_Road'] = df['Road_Condition'].isin(['Wet', 'Icy', 'Construction'])
        
        return df
        
    except Exception as e:
        print(f"Error in process_traffic_data: {str(e)}")
        # Return a minimal valid DataFrame if processing fails
        return pd.DataFrame({
            'Date': [datetime.now()],
            'Time': ['00:00'],
            'Hour': [0],
            'Location': ['Unknown'],
            'Is_Freeway': [False],
            'Freeway_Name': ['Local Road'],
            'Risk_Score': [0.0],
            'Is_Rush_Hour': [False],
            'Is_Night': [False],
            'Has_Adverse_Weather': [False],
            'Has_Adverse_Road': [False]
        })

def extract_freeway_name(location):
    """
    Extract freeway name from location string.
    
    Args:
        location (str): Location string
        
    Returns:
        str: Freeway name or 'Non-Freeway'
    """
    if pd.isna(location):
        return 'Non-Freeway'
    
    location = str(location).upper()
    
    if '101' in location:
        return 'US-101'
    elif '280' in location:
        return 'CA-280'
    elif '680' in location:
        return 'CA-680'
    elif '87' in location:
        return 'CA-87'
    else:
        return 'Non-Freeway'

def calculate_risk_score(df):
    """Calculate risk scores based on various factors."""
    try:
        # Initialize base risk score
        risk_scores = np.zeros(len(df))
        factors_used = 0
        
        # Time-based risks (rush hours and night driving)
        if 'Hour' in df.columns:
            # Rush hour multiplier (1.5x risk during rush hours)
            rush_hour_mask = df['Hour'].apply(lambda x: (x >= 7 and x <= 9) or (x >= 16 and x <= 18))
            risk_scores[rush_hour_mask] += 0.5
            factors_used += 1
            
            # Night driving multiplier (1.3x risk at night)
            night_mask = df['Hour'].apply(lambda x: x < 6 or x > 18)
            risk_scores[night_mask] += 0.3
            factors_used += 1
        
        # Weather conditions
        if 'Weather' in df.columns:
            weather_risk = {
                'Clear': 0.0,
                'Rain': 0.4,
                'Fog': 0.6,
                'Wind': 0.3
            }
            risk_scores += df['Weather'].map(weather_risk).fillna(0)
            factors_used += 1
        
        # Road conditions
        if 'Road_Condition' in df.columns:
            road_risk = {
                'Dry': 0.0,
                'Wet': 0.3,
                'Icy': 0.8,
                'Construction': 0.4
            }
            risk_scores += df['Road_Condition'].map(road_risk).fillna(0)
            factors_used += 1
        
        # Freeway vs local road
        if 'Is_Freeway' in df.columns:
            # Freeways generally have higher risk due to higher speeds
            risk_scores[df['Is_Freeway']] += 0.2
            factors_used += 1
        
        # Vehicle damage severity
        if 'VehicleDamage' in df.columns:
            damage_risk = {
                'Minor': 0.1,
                'Moderate': 0.3,
                'Severe': 0.6
            }
            risk_scores += df['VehicleDamage'].map(damage_risk).fillna(0)
            factors_used += 1
        
        # Movement preceding collision
        if 'MovementPrecedingCollision' in df.columns:
            movement_risk = {
                'Proceeding Straight': 0.1,
                'Making Left Turn': 0.4,
                'Changing Lanes': 0.3,
                'Backing': 0.2,
                'Parking Maneuver': 0.1,
                'Turning Right': 0.2,
                'Merging': 0.3,
                'Stopping': 0.1,
                'Overtaking': 0.4,
                'Passing': 0.3,
                'Unknown': 0.2
            }
            risk_scores += df['MovementPrecedingCollision'].map(movement_risk).fillna(0)
            factors_used += 1
        
        # Number of vehicles involved
        if 'Vehicles_Involved' in df.columns:
            # More vehicles = higher risk
            risk_scores += (df['Vehicles_Involved'] - 1) * 0.2
            factors_used += 1
        
        # Injuries
        if 'Injuries' in df.columns:
            # More injuries = higher risk
            risk_scores += df['Injuries'] * 0.15
            factors_used += 1
        
        # Normalize risk scores to 0-1 range
        if factors_used > 0:
            risk_scores = risk_scores / factors_used
            # Cap at 1.0
            risk_scores = np.minimum(risk_scores, 1.0)
        
        return risk_scores
    
    except Exception as e:
        print(f"Error in calculate_risk_score: {str(e)}")
        # Return a default risk score of 0.5 if calculation fails
        return np.ones(len(df)) * 0.5

def format_time_12hr(hour: int) -> str:
    """
    Format hour in 12-hour PST format.
    
    Args:
        hour: Hour in 24-hour format (0-23)
        
    Returns:
        Formatted time string in 12-hour PST format
    """
    try:
        pst = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pst)
        time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        return time.strftime("%I:%M %p PST")
    except Exception as e:
        print(f"Error formatting time: {str(e)}")
        return f"{hour:02d}:00"

def analyze_freeway_safety(df):
    """Analyze freeway safety and provide recommendations."""
    try:
        # Filter for freeway crashes
        freeway_df = df[df['Is_Freeway'] == True].copy()
        
        if len(freeway_df) == 0:
            return {
                'safest_freeway': None,
                'safest_freeway_risk': 0.0,
                'freeway_risk_scores': {},
                'freeway_crash_counts': {},
                'recommendations': ['No freeway data available for analysis.']
            }
        
        # Calculate risk scores by freeway
        if 'Injuries' in freeway_df.columns:
            freeway_risk = freeway_df.groupby('Freeway_Name').agg({
                'Risk_Score': 'mean',
                'Injuries': 'sum'  # Use Injuries if available
            }).reset_index()
        else:
            # If Injuries column doesn't exist, just use Risk_Score
            freeway_risk = freeway_df.groupby('Freeway_Name').agg({
                'Risk_Score': 'mean'
            }).reset_index()
            # Add a dummy Injuries column with zeros
            freeway_risk['Injuries'] = 0
        
        # Sort by risk score (lower is better)
        freeway_risk = freeway_risk.sort_values('Risk_Score')
        
        # Get safest freeway
        safest_freeway = freeway_risk.iloc[0]['Freeway_Name']
        safest_freeway_risk = freeway_risk.iloc[0]['Risk_Score']
        
        # Create risk score dictionary
        risk_scores = dict(zip(freeway_risk['Freeway_Name'], 
                             freeway_risk['Risk_Score'].round(2)))
        
        # Create crash count dictionary using Injuries as a proxy for crash severity
        crash_counts = dict(zip(freeway_risk['Freeway_Name'], 
                              freeway_risk['Injuries']))
        
        # Generate recommendations
        recommendations = [
            f"Consider using {safest_freeway} as it has the lowest risk score of {risk_scores[safest_freeway]:.2f}",
            f"Be extra cautious on {freeway_risk.iloc[-1]['Freeway_Name']} which has the highest risk score of {freeway_risk.iloc[-1]['Risk_Score']:.2f}"
        ]
        
        # Add time-based recommendations
        morning_freeway = freeway_df[freeway_df['Hour'].between(6, 9)].groupby('Freeway_Name')['Risk_Score'].mean().idxmin()
        evening_freeway = freeway_df[freeway_df['Hour'].between(16, 19)].groupby('Freeway_Name')['Risk_Score'].mean().idxmin()
        
        recommendations.extend([
            f"For morning commute (6-9 AM), {morning_freeway} tends to be safer",
            f"For evening commute (4-7 PM), {evening_freeway} tends to be safer"
        ])
        
        return {
            'safest_freeway': safest_freeway,
            'safest_freeway_risk': safest_freeway_risk,
            'freeway_risk_scores': risk_scores,
            'freeway_crash_counts': crash_counts,
            'recommendations': recommendations
        }
    except Exception as e:
        print(f"Error in freeway analysis: {str(e)}")
        return {
            'safest_freeway': None,
            'safest_freeway_risk': 0.0,
            'freeway_risk_scores': {},
            'freeway_crash_counts': {},
            'recommendations': ['Error analyzing freeway data.']
        }

def generate_commute_recommendations(df, route_type='Freeway', avoid_construction=True, prefer_scenic=False, avoid_high_risk=True):
    """Generate specific commute recommendations based on historical data and preferences."""
    try:
        recommendations = []
        
        # Define major office locations in San Jose
        office_locations = {
            'Downtown SJ': {'lat': 37.3382, 'lon': -121.8863},
            'North SJ': {'lat': 37.4082, 'lon': -121.9163},
            'South SJ': {'lat': 37.2582, 'lon': -121.8563},
            'East SJ': {'lat': 37.3382, 'lon': -121.7563},
            'West SJ': {'lat': 37.3382, 'lon': -122.0163}
        }
        
        # Filter for relevant time periods
        morning_data = df[df['Hour'].between(5, 10)]
        evening_data = df[df['Hour'].between(15, 20)]
        
        # Generate time-based recommendations
        if not morning_data.empty:
            # Find safest morning commute time
            morning_risks = morning_data.groupby('Hour')['Risk_Score'].mean()
            safest_morning = morning_risks.idxmin()
            recommendations.append(f"🕒 Best morning commute time: {format_time_12hr(safest_morning)}")
        
        if not evening_data.empty:
            # Find safest evening commute time
            evening_risks = evening_data.groupby('Hour')['Risk_Score'].mean()
            safest_evening = evening_risks.idxmin()
            recommendations.append(f"🕒 Best evening commute time: {format_time_12hr(safest_evening)}")
        
        # Weather-based recommendations
        if 'Weather' in df.columns:
            weather_risks = df.groupby('Weather')['Risk_Score'].mean()
            if 'Rain' in weather_risks.index and weather_risks['Rain'] > 0.4:
                recommendations.append("🌧️ Consider leaving earlier during rainy conditions")
            if 'Fog' in weather_risks.index and weather_risks['Fog'] > 0.5:
                recommendations.append("🌫️ Allow extra time for foggy conditions")
        
        # Road condition recommendations
        if 'Road_Condition' in df.columns:
            road_risks = df.groupby('Road_Condition')['Risk_Score'].mean()
            if 'Wet' in road_risks.index and road_risks['Wet'] > 0.4:
                recommendations.append("🚗 Exercise caution on wet roads")
            if 'Construction' in road_risks.index and road_risks['Construction'] > 0.4:
                recommendations.append("🚧 Expect delays in construction zones")
        
        # Movement-based recommendations
        if 'MovementPrecedingCollision' in df.columns:
            movement_risks = df.groupby('MovementPrecedingCollision')['Risk_Score'].mean()
            high_risk_movements = movement_risks[movement_risks > 0.4].index
            if len(high_risk_movements) > 0:
                recommendations.append("⚠️ Be extra cautious when: " + ", ".join(high_risk_movements))
        
        # Route-specific recommendations
        if 'Freeway_Name' in df.columns:
            freeway_risks = df.groupby('Freeway_Name')['Risk_Score'].mean()
            safest_freeway = freeway_risks.idxmin()
            recommendations.append(f"🛣️ Consider using {safest_freeway} for lowest risk")
        
        # Office location-specific recommendations
        for office, coords in office_locations.items():
            # Calculate average risk score for incidents near this office
            office_data = df[
                (df['Latitude'].between(coords['lat'] - 0.01, coords['lat'] + 0.01)) &
                (df['Longitude'].between(coords['lon'] - 0.01, coords['lon'] + 0.01))
            ]
            if not office_data.empty:
                avg_risk = office_data['Risk_Score'].mean()
                if avg_risk > 0.6:
                    recommendations.append(f"⚠️ High risk area around {office}")
                elif avg_risk < 0.3:
                    recommendations.append(f"✅ Lower risk area around {office}")
        
        # Add general safety tips
        recommendations.extend([
            "🚗 Maintain safe following distance",
            "📱 Avoid distracted driving",
            "🚦 Obey traffic signals and signs",
            "🌙 Use headlights during low visibility",
            "🔄 Check blind spots before changing lanes"
        ])
        
        return recommendations
    
    except Exception as e:
        print(f"Error in generate_commute_recommendations: {str(e)}")
        return ["Unable to generate recommendations at this time. Please try again later."]

def generate_data_summary(df):
    """
    Generate a comprehensive summary of the traffic data for AI analysis.
    """
    summary = {
        'total_crashes': len(df),
        'severity_distribution': df['Severity'].value_counts().to_dict(),
        'time_distribution': {
            'hourly': df.groupby('Hour').size().to_dict(),
            'commute_periods': df.groupby('Commute_Period').size().to_dict()
        },
        'location_analysis': {
            'highway_crashes': len(df[df['Is_Highway']]) if 'Is_Highway' in df.columns else 0,
            'intersection_crashes': len(df[df['Is_Intersection']]) if 'Is_Intersection' in df.columns else 0,
            'top_locations': df['Location'].value_counts().head(5).to_dict(),
            'freeway_crashes': len(df[df['Is_Freeway']]),
            'freeway_distribution': df['Freeway_Name'].value_counts().to_dict()
        },
        'impact_metrics': {
            'total_injuries': df['Injuries'].sum(),
            'total_fatalities': df['Fatalities'].sum(),
            'avg_vehicles_involved': df['Vehicles_Involved'].mean()
        },
        'environmental_factors': {
            'weather_distribution': df['Weather'].value_counts().to_dict(),
            'road_conditions': df['Road_Condition'].value_counts().to_dict(),
            'light_conditions': df['Light_Condition'].value_counts().to_dict()
        },
        'collision_patterns': df['Collision_Type'].value_counts().to_dict()
    }
    
    return summary 

def fetch_live_traffic_data(api_key: str) -> pd.DataFrame:
    """
    Fetch live traffic data from multiple sources.
    
    Args:
        api_key: API key for traffic data services
        
    Returns:
        DataFrame containing live traffic incidents
    """
    try:
        # For development, return mock data if API key is not set
        if api_key == "YOUR_API_KEY":
            return generate_mock_traffic_data()
            
        # Example: Fetch from Caltrans API
        caltrans_url = f"https://api.dot.ca.gov/v1/incidents?api_key={api_key}"
        response = requests.get(caltrans_url, timeout=10)
        response.raise_for_status()  # Raise exception for bad status codes
        incidents = response.json()
        
        # Process incidents into DataFrame
        live_data = []
        for incident in incidents:
            live_data.append({
                'Date': datetime.now(),
                'Location': incident.get('location', 'Unknown'),
                'Latitude': incident.get('latitude'),
                'Longitude': incident.get('longitude'),
                'Is_Freeway': incident.get('is_freeway', False),
                'Severity': incident.get('severity', 'Unknown'),
                'Description': incident.get('description', ''),
                'Last_Updated': incident.get('last_updated', datetime.now()),
                'Data_Source': 'Caltrans',
                'Confidence_Score': incident.get('confidence_score', 0.8)
            })
        
        return pd.DataFrame(live_data)
    except Exception as e:
        print(f"Error fetching live traffic data: {str(e)}")
        return generate_mock_traffic_data()

def generate_mock_traffic_data(n_samples: int = 10) -> pd.DataFrame:
    """
    Generate mock traffic data for development and testing.
    
    Args:
        n_samples: Number of mock incidents to generate
        
    Returns:
        DataFrame containing mock traffic incidents
    """
    # San Jose coordinates
    sj_lat = 37.3382
    sj_lon = -121.8863
    
    # Generate random coordinates around San Jose
    lats = np.random.normal(sj_lat, 0.05, n_samples)
    lons = np.random.normal(sj_lon, 0.05, n_samples)
    
    # Generate mock data
    mock_data = {
        'Date': [datetime.now() - timedelta(hours=np.random.randint(0, 24)) for _ in range(n_samples)],
        'Location': [f"Location_{i}" for i in range(n_samples)],
        'Latitude': lats,
        'Longitude': lons,
        'Is_Freeway': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        'Severity': np.random.choice(['Minor', 'Moderate', 'Severe'], n_samples, p=[0.6, 0.3, 0.1]),
        'Description': [f"Mock incident {i}" for i in range(n_samples)],
        'Last_Updated': [datetime.now() for _ in range(n_samples)],
        'Data_Source': ['Mock'] * n_samples,
        'Confidence_Score': np.random.uniform(0.7, 1.0, n_samples)
    }
    
    return pd.DataFrame(mock_data)

def fetch_weather_data() -> Dict[str, Any]:
    """
    Fetch current weather data for San Jose
    Returns a dictionary with weather information
    """
    try:
        # Simulated weather data for now
        # In production, this would call a weather API
        weather_conditions = ['Clear', 'Partly Cloudy', 'Cloudy', 'Light Rain', 'Rain']
        current_weather = {
            'condition': random.choice(weather_conditions),
            'temperature': round(random.uniform(50, 85), 1),
            'wind_speed': round(random.uniform(0, 20), 1),
            'precipitation': round(random.uniform(0, 100), 0)
        }
        return current_weather
    except Exception as e:
        logger.error(f"Error fetching weather data: {str(e)}")
        return {}

def fetch_special_events() -> List[Dict[str, Any]]:
    """
    Fetch special events data for San Jose
    Returns a list of dictionaries containing event information
    """
    try:
        # Simulated events data for now
        # In production, this would call an events API
        event_types = ['Construction', 'Sports Event', 'Concert', 'Road Work', 'Festival']
        impact_levels = ['Low', 'Medium', 'High']
        locations = ['Downtown', 'North San Jose', 'South San Jose', 'East San Jose', 'West San Jose']
        
        num_events = random.randint(0, 3)
        events = []
        
        for _ in range(num_events):
            event = {
                'name': f"{random.choice(event_types)}",
                'location': random.choice(locations),
                'time': format_time_12hr(random.randint(0, 23)),
                'impact': random.choice(impact_levels)
            }
            events.append(event)
        
        return events
    except Exception as e:
        logger.error(f"Error fetching events data: {str(e)}")
        print(f"Error fetching events data: {str(e)}")
        return None

def generate_mock_weather_data() -> Dict:
    """
    Generate mock weather data for development and testing.
    
    Returns:
        Dictionary containing mock weather data
    """
    conditions = ['Clear', 'Clouds', 'Rain', 'Snow', 'Thunderstorm', 'Fog']
    return {
        'condition': np.random.choice(conditions),
        'description': 'Mock weather data for development',
        'temperature': np.random.uniform(280, 300),  # Kelvin
        'visibility': np.random.uniform(5000, 10000),
        'wind_speed': np.random.uniform(0, 20),
        'last_updated': datetime.now()
    }

def calculate_event_impact(event: Dict) -> float:
    """
    Calculate the traffic impact score of a special event.
    
    Args:
        event: Event data dictionary
        
    Returns:
        Impact score between 0 and 1
    """
    try:
        # Factors that influence impact
        attendee_factor = min(event.get('attendees', 0) / 10000, 1.0)  # Normalize by 10,000 attendees
        duration_factor = calculate_duration_factor(event.get('start_time', ''), event.get('end_time', ''))
        venue_factor = calculate_venue_factor(event.get('venue', ''))
        
        # Weighted combination of factors
        impact_score = (
            0.4 * attendee_factor +
            0.3 * duration_factor +
            0.3 * venue_factor
        )
        
        return min(max(impact_score, 0), 1)  # Ensure score is between 0 and 1
    except Exception as e:
        print(f"Error calculating event impact: {str(e)}")
        return 0.0

def calculate_dynamic_risk_score(df: pd.DataFrame, 
                              weather_data: Optional[Dict] = None,
                              special_events: Optional[List[Dict]] = None) -> pd.Series:
    """
    Calculate dynamic risk scores incorporating real-time data.
    
    Args:
        df: DataFrame containing crash data
        weather_data: Current weather conditions
        special_events: List of special events
        
    Returns:
        Series containing updated risk scores
    """
    try:
        # Initialize risk score array
        risk_scores = np.zeros(len(df))
        factor_count = 0
        
        # Base risk factors (existing calculation)
        if 'VehicleDamage' in df.columns:
            risk_scores += df['VehicleDamage'].map({
                'Severe': 1.0,
                'Moderate': 0.7,
                'Minor': 0.4
            }).fillna(0.5)
            factor_count += 1
        
        # Weather impact
        if weather_data:
            weather_impact = calculate_weather_impact(weather_data)
            risk_scores += weather_impact
            factor_count += 1
        
        # Special events impact
        if special_events:
            event_impact = calculate_events_impact(df, special_events)
            risk_scores += event_impact
            factor_count += 1
        
        # Time-based adjustments
        if 'Hour' in df.columns:
            time_impact = calculate_time_impact(df['Hour'])
            risk_scores += time_impact
            factor_count += 1
        
        # Normalize by number of factors used
        if factor_count > 0:
            risk_scores = risk_scores / factor_count
        
        # Ensure scores are between 0 and 1
        risk_scores = np.clip(risk_scores, 0, 1)
        
        return pd.Series(risk_scores, index=df.index)
    
    except Exception as e:
        print(f"Error calculating dynamic risk score: {str(e)}")
        return pd.Series(0.5, index=df.index)

def calculate_weather_impact(weather_data: Dict) -> float:
    """
    Calculate risk impact of current weather conditions.
    
    Args:
        weather_data: Current weather data
        
    Returns:
        Weather impact score between 0 and 1
    """
    try:
        impact = 0.0
        
        # Weather condition impact
        condition_impacts = {
            'Clear': 0.0,
            'Clouds': 0.1,
            'Rain': 0.3,
            'Snow': 0.5,
            'Thunderstorm': 0.7,
            'Fog': 0.6
        }
        impact += condition_impacts.get(weather_data.get('condition', 'Clear'), 0.0)
        
        # Visibility impact
        visibility = weather_data.get('visibility', 10000)
        if visibility < 1000:
            impact += 0.4
        elif visibility < 5000:
            impact += 0.2
        
        # Wind impact
        wind_speed = weather_data.get('wind_speed', 0)
        if wind_speed > 20:
            impact += 0.3
        elif wind_speed > 10:
            impact += 0.1
        
        return min(impact, 1.0)
    
    except Exception as e:
        print(f"Error calculating weather impact: {str(e)}")
        return 0.0

def calculate_events_impact(df: pd.DataFrame, events: List[Dict]) -> np.ndarray:
    """
    Calculate risk impact of special events.
    
    Args:
        df: DataFrame containing crash data
        events: List of special events
        
    Returns:
        Array of event impact scores
    """
    try:
        impact_scores = np.zeros(len(df))
        
        for idx, row in df.iterrows():
            # Calculate distance to each event
            for event in events:
                event_venue = event.get('venue', '')
                if event_venue:
                    # Simplified distance calculation (you should use proper geocoding)
                    distance = calculate_distance(
                        row['Latitude'], row['Longitude'],
                        event.get('venue_lat', 0), event.get('venue_lon', 0)
                    )
                    
                    # Impact decreases with distance
                    if distance < 1:  # Within 1 mile
                        impact_scores[idx] += event.get('impact_score', 0) * 0.8
                    elif distance < 3:  # Within 3 miles
                        impact_scores[idx] += event.get('impact_score', 0) * 0.5
                    elif distance < 5:  # Within 5 miles
                        impact_scores[idx] += event.get('impact_score', 0) * 0.2
        
        return np.clip(impact_scores, 0, 1)
    
    except Exception as e:
        print(f"Error calculating events impact: {str(e)}")
        return np.zeros(len(df))

def calculate_time_impact(hours: pd.Series) -> np.ndarray:
    """
    Calculate time-based risk impact.
    
    Args:
        hours: Series of hours (0-23)
        
    Returns:
        Array of time impact scores
    """
    try:
        impact_scores = np.zeros(len(hours))
        
        # Rush hour impact
        morning_rush = (hours >= 7) & (hours <= 9)
        evening_rush = (hours >= 16) & (hours <= 18)
        impact_scores[morning_rush] += 0.3
        impact_scores[evening_rush] += 0.3
        
        # Night impact
        night_hours = (hours < 6) | (hours > 18)
        impact_scores[night_hours] += 0.2
        
        return np.clip(impact_scores, 0, 1)
    
    except Exception as e:
        print(f"Error calculating time impact: {str(e)}")
        return np.zeros(len(hours))

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points using Haversine formula.
    
    Args:
        lat1: Latitude of first point
        lon1: Longitude of first point
        lat2: Latitude of second point
        lon2: Longitude of second point
        
    Returns:
        Distance in miles
    """
    try:
        R = 3959  # Earth's radius in miles
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    except Exception as e:
        print(f"Error calculating distance: {str(e)}")
        return float('inf')

def generate_enhanced_route_recommendations(
    df: pd.DataFrame,
    start_location: str,
    end_location: str,
    preferred_time: Optional[datetime] = None
) -> Dict:
    """
    Generate enhanced route recommendations with safety scores and alternatives.
    
    Args:
        df (pd.DataFrame): Traffic data
        start_location (str): Starting location
        end_location (str): Destination
        preferred_time (datetime, optional): Preferred departure time
        
    Returns:
        Dict: Route recommendations with safety information
    """
    try:
        # Validate inputs
        if not start_location or not end_location:
            return {
                'error': 'Please provide both start and end locations',
                'current_conditions': get_current_conditions(df)
            }
            
        # Define route options with enhanced safety information
        routes = {
            'Freeway': {
                'description': 'Primary freeway route with higher speed limits',
                'safety_tips': [
                    'Maintain safe following distance (3-4 seconds)',
                    'Watch for sudden lane changes and merging traffic',
                    'Use turn signals when changing lanes',
                    'Keep right except when passing',
                    'Check blind spots before changing lanes'
                ],
                'risk_factor': 1.2
            },
            'Local Roads': {
                'description': 'Alternative local route with lower speed limits',
                'safety_tips': [
                    'Watch for pedestrians and cyclists',
                    'Be cautious at intersections and crosswalks',
                    'Observe local speed limits',
                    'Yield to emergency vehicles',
                    'Watch for school zones during school hours'
                ],
                'risk_factor': 0.8
            },
            'Mixed Route': {
                'description': 'Combination of freeway and local roads',
                'safety_tips': [
                    'Plan transitions between road types',
                    'Be prepared for different speed limits',
                    'Watch for changing traffic patterns',
                    'Use navigation app for real-time updates',
                    'Allow extra time for route changes'
                ],
                'risk_factor': 1.0
            }
        }
        
        # Get current conditions
        current_conditions = get_current_conditions(df)
        
        # Calculate route-specific risk scores
        route_risks = {}
        for route_type, route_info in routes.items():
            base_risk = predict_risk_score(
                location=start_location,
                hour=current_conditions['hour'],
                weather=current_conditions['weather'],
                road_condition=current_conditions['road_condition']
            )
            
            # Adjust risk based on route type and current conditions
            adjusted_risk = base_risk * route_info['risk_factor']
            
            # Add time-based adjustments
            if preferred_time:
                hour = preferred_time.hour
                if hour in [7, 8, 9, 16, 17, 18]:  # Rush hours
                    adjusted_risk *= 1.2
                elif hour < 6 or hour > 18:  # Night
                    adjusted_risk *= 1.1
                    
            route_risks[route_type] = min(1.0, adjusted_risk)
        
        # Generate recommendations
        recommendations = {
            'current_conditions': current_conditions,
            'routes': {}
        }
        
        # Add route details with enhanced information
        for route_type, risk_score in route_risks.items():
            route_info = routes[route_type]
            estimated_time = calculate_estimated_time(risk_score, route_type)
            
            recommendations['routes'][route_type] = {
                'risk_score': risk_score,
                'description': route_info['description'],
                'safety_tips': route_info['safety_tips'],
                'estimated_time': estimated_time,
                'recommendation': get_route_recommendation(risk_score),
                'alternatives': get_alternative_routes(route_type, route_risks)
            }
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating route recommendations: {str(e)}")
        return {
            'error': 'Unable to generate route recommendations. Please try again.',
            'current_conditions': get_current_conditions(df)
        }

def get_current_conditions(df: pd.DataFrame) -> Dict:
    """
    Get current conditions from the dataset.
    
    Args:
        df (pd.DataFrame): Traffic data
        
    Returns:
        Dict: Current conditions including weather, road condition, and time
    """
    try:
        current_hour = datetime.now().hour
        current_weather = df['Weather'].mode().iloc[0] if not df['Weather'].empty else 'Unknown'
        current_road = df['Road_Condition'].mode().iloc[0] if not df['Road_Condition'].empty else 'Unknown'
        
        return {
            'weather': current_weather,
            'road_condition': current_road,
            'hour': current_hour,
            'time': datetime.now().strftime('%I:%M %p PST')
        }
    except Exception as e:
        print(f"Error getting current conditions: {str(e)}")
        return {
            'weather': 'Unknown',
            'road_condition': 'Unknown',
            'hour': datetime.now().hour,
            'time': datetime.now().strftime('%I:%M %p PST')
        }

def calculate_estimated_time(risk_score: float, route_type: str) -> str:
    """
    Calculate estimated travel time based on risk score and route type.
    
    Args:
        risk_score (float): Risk score for the route
        route_type (str): Type of route
        
    Returns:
        str: Estimated travel time
    """
    base_times = {
        'Freeway': 30,
        'Local Roads': 45,
        'Mixed Route': 35
    }
    
    base_time = base_times.get(route_type, 30)
    adjusted_time = base_time * (1 + risk_score)
    
    return f"{int(adjusted_time)} minutes"

def get_route_recommendation(risk_score: float) -> str:
    """
    Get route recommendation based on risk score.
    
    Args:
        risk_score (float): Risk score for the route
        
    Returns:
        str: Route recommendation
    """
    if risk_score < 0.3:
        return 'Highly Recommended'
    elif risk_score < 0.5:
        return 'Recommended'
    elif risk_score < 0.7:
        return 'Use Caution'
    else:
        return 'High Risk - Consider Alternative'

def get_alternative_routes(current_route: str, route_risks: Dict[str, float]) -> List[str]:
    """
    Get alternative route suggestions based on current route and risk scores.
    
    Args:
        current_route (str): Current route type
        route_risks (Dict[str, float]): Risk scores for all routes
        
    Returns:
        List[str]: List of alternative route suggestions
    """
    alternatives = []
    for route, risk in route_risks.items():
        if route != current_route and risk < route_risks[current_route]:
            alternatives.append(f"Consider {route} route (lower risk)")
    return alternatives

def predict_risk_score(location: str, hour: int, weather: str, road_condition: str) -> float:
    """
    Predict risk score for a given location and conditions using machine learning.
    
    Args:
        location (str): Location name
        hour (int): Hour of day (0-23)
        weather (str): Weather condition
        road_condition (str): Road condition
        
    Returns:
        float: Predicted risk score between 0 and 1
    """
    try:
        model_path = 'models/risk_predictor.joblib'
        scaler_path = 'models/risk_scaler.joblib'
        
        # Check if model files exist
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            print("Model files not found. Using default risk prediction.")
            return calculate_default_risk_score(hour, weather, road_condition)
        
        # Load the trained model
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Prepare features
        features = pd.DataFrame({
            'hour': [hour],
            'is_rush_hour': [1 if hour in [7, 8, 9, 16, 17, 18] else 0],
            'is_night': [1 if hour < 6 or hour > 18 else 0],
            'is_weekend': [1 if datetime.now().weekday() >= 5 else 0],
            'weather_rainy': [1 if weather == 'Rainy' else 0],
            'weather_foggy': [1 if weather == 'Foggy' else 0],
            'road_wet': [1 if road_condition == 'Wet' else 0],
            'road_icy': [1 if road_condition == 'Icy' else 0]
        })
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        risk_score = model.predict(features_scaled)[0]
        return max(0, min(1, risk_score))  # Ensure score is between 0 and 1
        
    except Exception as e:
        print(f"Error in risk prediction: {str(e)}")
        return calculate_default_risk_score(hour, weather, road_condition)

def calculate_default_risk_score(hour: int, weather: str, road_condition: str) -> float:
    """
    Calculate a default risk score when the ML model is not available.
    
    Args:
        hour (int): Hour of day (0-23)
        weather (str): Weather condition
        road_condition (str): Road condition
        
    Returns:
        float: Default risk score between 0 and 1
    """
    # Base risk score
    risk_score = 0.5
    
    # Time-based adjustments
    if hour in [7, 8, 9, 16, 17, 18]:  # Rush hours
        risk_score += 0.2
    elif hour < 6 or hour > 18:  # Night
        risk_score += 0.1
        
    # Weather adjustments
    weather_risk = {
        'Clear': 0.0,
        'Clouds': 0.1,
        'Rain': 0.3,
        'Snow': 0.5,
        'Thunderstorm': 0.7,
        'Fog': 0.6
    }
    risk_score += weather_risk.get(weather, 0.0)
    
    # Road condition adjustments
    road_risk = {
        'Dry': 0.0,
        'Wet': 0.3,
        'Icy': 0.8,
        'Construction': 0.4
    }
    risk_score += road_risk.get(road_condition, 0.0)
    
    # Normalize and ensure score is between 0 and 1
    return max(0, min(1, risk_score))

def train_risk_predictor(df: pd.DataFrame) -> None:
    """
    Train a machine learning model to predict risk scores.
    
    Args:
        df (pd.DataFrame): Historical traffic data
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Prepare features
        features = pd.DataFrame({
            'hour': df['Hour'],
            'is_rush_hour': df['Hour'].apply(lambda x: 1 if x in [7, 8, 9, 16, 17, 18] else 0),
            'is_night': df['Hour'].apply(lambda x: 1 if x < 6 or x > 18 else 0),
            'is_weekend': df['Date'].apply(lambda x: 1 if x.weekday() >= 5 else 0),
            'weather_rainy': (df['Weather'] == 'Rainy').astype(int),
            'weather_foggy': (df['Weather'] == 'Foggy').astype(int),
            'road_wet': (df['Road_Condition'] == 'Wet').astype(int),
            'road_icy': (df['Road_Condition'] == 'Icy').astype(int)
        })
        
        # Prepare target
        target = df['Risk_Score']
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, target)
        
        # Save model and scaler
        joblib.dump(model, 'models/risk_predictor.joblib')
        joblib.dump(scaler, 'models/risk_scaler.joblib')
        
        print("Risk prediction model trained and saved successfully.")
        
    except Exception as e:
        print(f"Error training risk predictor: {str(e)}")
        # Create a default model if training fails
        create_default_model()

def create_default_model() -> None:
    """
    Create a default risk prediction model when training fails.
    """
    try:
        # Create a simple model with default parameters
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        scaler = StandardScaler()
        
        # Create dummy data
        X = np.random.rand(100, 8)  # 8 features
        y = np.random.rand(100)     # Target values
        
        # Fit the model
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        # Save the default model
        joblib.dump(model, 'models/risk_predictor.joblib')
        joblib.dump(scaler, 'models/risk_scaler.joblib')
        
        print("Default risk prediction model created.")
        
    except Exception as e:
        print(f"Error creating default model: {str(e)}")

def get_public_transit_info(start_location: str, end_location: str) -> Dict:
    """
    Get public transit information for a route.
    
    Args:
        start_location: Starting location
        end_location: Destination
        
    Returns:
        Dictionary containing transit information
    """
    try:
        # This would typically call a transit API
        # For now, return mock data
        return {
            'light_rail': {
                'frequency': '15-20 minutes',
                'duration': '45 minutes',
                'cost': 2.50,
                'transfers': 1
            },
            'bus': {
                'frequency': '30 minutes',
                'duration': '60 minutes',
                'cost': 2.25,
                'transfers': 2
            },
            'express_bus': {
                'frequency': 'Peak hours only',
                'duration': '35 minutes',
                'cost': 3.00,
                'transfers': 0
            }
        }
    except Exception as e:
        print(f"Error getting transit info: {str(e)}")
        return {}

def get_parking_info(location: str) -> Dict:
    """
    Get parking information for a location.
    
    Args:
        location: Location to get parking info for
        
    Returns:
        Dictionary containing parking information
    """
    try:
        # This would typically call a parking API
        # For now, return mock data
        return {
            'street_parking': {
                'availability': 'Limited',
                'cost': '$2/hour',
                'time_limit': '2 hours'
            },
            'garages': {
                'availability': 'Good',
                'cost': '$5/hour',
                'time_limit': 'No limit'
            },
            'park_and_ride': {
                'availability': 'Excellent',
                'cost': '$5/day',
                'time_limit': '24 hours'
            }
        }
    except Exception as e:
        print(f"Error getting parking info: {str(e)}")
        return {}

def calculate_eco_impact(distance: float, mode: str) -> Dict:
    """
    Calculate environmental impact of a trip.
    
    Args:
        distance: Distance in miles
        mode: Transportation mode
        
    Returns:
        Dictionary containing environmental impact metrics
    """
    try:
        # CO2 emissions in kg per mile
        emissions = {
            'Drive': 0.404,  # Average car
            'Public Transit': 0.104,  # Bus
            'Bike': 0,
            'Walk': 0
        }
        
        co2 = distance * emissions.get(mode, 0.404)
        
        return {
            'co2_emissions': co2,
            'trees_needed': co2 / 22,  # One tree absorbs ~22kg CO2 per year
            'equivalent_miles': co2 * 2.5  # Equivalent to driving this many miles
        }
    except Exception as e:
        print(f"Error calculating eco impact: {str(e)}")
        return {}

def get_historical_traffic_patterns(location: str, day_of_week: int) -> Dict:
    """
    Get historical traffic patterns for a location.
    
    Args:
        location: Location to analyze
        day_of_week: Day of week (0-6, Monday-Sunday)
        
    Returns:
        Dictionary containing traffic pattern information
    """
    try:
        # This would typically analyze historical data
        # For now, return mock data
        patterns = {
            0: {  # Monday
                'peak_hours': ['7:00-9:00', '16:00-18:00'],
                'best_times': ['10:00-15:00'],
                'avg_speed': 35
            },
            1: {  # Tuesday
                'peak_hours': ['7:00-9:00', '16:00-18:00'],
                'best_times': ['10:00-15:00'],
                'avg_speed': 35
            },
            2: {  # Wednesday
                'peak_hours': ['7:00-9:00', '16:00-18:00'],
                'best_times': ['10:00-15:00'],
                'avg_speed': 35
            },
            3: {  # Thursday
                'peak_hours': ['7:00-9:00', '16:00-18:00'],
                'best_times': ['10:00-15:00'],
                'avg_speed': 35
            },
            4: {  # Friday
                'peak_hours': ['7:00-9:00', '15:00-19:00'],
                'best_times': ['10:00-14:00'],
                'avg_speed': 30
            },
            5: {  # Saturday
                'peak_hours': ['11:00-14:00', '16:00-19:00'],
                'best_times': ['9:00-11:00', '14:00-16:00'],
                'avg_speed': 40
            },
            6: {  # Sunday
                'peak_hours': ['11:00-14:00', '16:00-19:00'],
                'best_times': ['9:00-11:00', '14:00-16:00'],
                'avg_speed': 45
            }
        }
        
        return patterns.get(day_of_week, patterns[0])
    except Exception as e:
        print(f"Error getting traffic patterns: {str(e)}")
        return {} 