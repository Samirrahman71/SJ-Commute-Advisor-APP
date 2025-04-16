import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import openai

# Load environment variables
load_dotenv()

class TrafficAnalysisAI:
    def __init__(self):
        """Initialize the AI analyzer with OpenAI API key"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        
        # Set the API key directly
        openai.api_key = api_key
        
        # We'll use the openai module directly instead of the client
        self.client = openai
        
    def analyze_crash_patterns(self, data_summary):
        """
        Analyze crash patterns using GPT-4 with a carefully engineered prompt.
        """
        prompt = f"""
        As a traffic safety analyst, analyze the following crash data from San Jose, CA:

        {data_summary}

        Please provide a comprehensive analysis covering:

        1. Temporal Patterns:
           - Identify peak crash times and days
           - Analyze seasonal trends
           - Highlight any unusual patterns

        2. Location Analysis:
           - Evaluate high-risk locations
           - Compare highway vs. intersection crashes
           - Identify geographical patterns

        3. Severity and Impact:
           - Analyze factors contributing to severe crashes
           - Evaluate injury and fatality patterns
           - Assess vehicle involvement patterns

        4. Environmental Factors:
           - Impact of weather conditions
           - Road condition effects
           - Lighting condition influence

        5. Recommendations:
           - Specific safety improvements
           - Traffic management suggestions
           - Infrastructure recommendations

        Format the response with clear sections and bullet points where appropriate.
        Focus on actionable insights and data-driven recommendations.
        """
        
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert traffic safety analyst with extensive experience in crash analysis and safety recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing crash patterns: {str(e)}"
    
    def predict_crash_risk(self, conditions):
        """
        Predict crash risk using GPT-4 with few-shot prompting.
        """
        prompt = f"""
        Based on the following traffic conditions, predict the risk level of a crash:

        Time: {conditions['time']}
        Location: {conditions['location']}
        Weather: {conditions['weather']}
        Road Condition: {conditions['road_condition']}
        Light Condition: {conditions['light_condition']}
        Vehicles Involved: {conditions['vehicles']}

        Consider these examples:
        1. Time: 8:00 AM, Location: HWY 101, Weather: Clear, Road: Dry, Light: Daylight, Vehicles: 2
           Risk: Moderate (Rush hour traffic)
        
        2. Time: 2:00 AM, Location: Local Street, Weather: Rain, Road: Wet, Light: Dark, Vehicles: 1
           Risk: High (Night + Rain + Wet roads)
        
        3. Time: 3:00 PM, Location: Intersection, Weather: Clear, Road: Dry, Light: Daylight, Vehicles: 2
           Risk: Low (Good conditions, moderate traffic)

        Provide:
        1. Risk Level (Low/Moderate/High)
        2. Confidence Score (0-100%)
        3. Key Risk Factors
        4. Safety Recommendations
        """
        
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a traffic safety expert specializing in risk assessment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error predicting crash risk: {str(e)}"
    
    def generate_safety_recommendations(self, data_summary, location_type):
        """
        Generate targeted safety recommendations based on data analysis.
        """
        prompt = f"""
        Based on the following crash data summary for {location_type} locations:

        {data_summary}

        Generate specific safety recommendations for:
        1. Infrastructure Improvements
        2. Traffic Management
        3. Driver Education
        4. Emergency Response
        5. Long-term Planning

        For each recommendation:
        - Explain the rationale
        - Estimate potential impact
        - Suggest implementation timeline
        - Identify key stakeholders

        Format as a detailed report with clear sections and prioritized recommendations.
        """
        
        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a traffic safety consultant specializing in infrastructure and policy recommendations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

def predict_crash_risk(df):
    """
    Analyze the dataset to identify high-risk factors and their impact on crash probability.
    
    Args:
        df (pd.DataFrame): Processed traffic crash data
        
    Returns:
        dict: Dictionary of risk factors and their percentage impact on crash risk
    """
    risk_factors = {}
    
    # Analyze weather impact
    weather_risk = df.groupby('Weather')['Risk_Score'].mean()
    weather_impact = ((weather_risk - weather_risk.mean()) / weather_risk.mean() * 100)
    for weather, impact in weather_impact.items():
        if abs(impact) > 5:  # Only include significant impacts
            risk_factors[f"Weather: {weather}"] = impact
    
    # Analyze road condition impact
    road_risk = df.groupby('Road_Condition')['Risk_Score'].mean()
    road_impact = ((road_risk - road_risk.mean()) / road_risk.mean() * 100)
    for condition, impact in road_impact.items():
        if abs(impact) > 5:
            risk_factors[f"Road Condition: {condition}"] = impact
    
    # Analyze time of day impact
    df['Is_Night'] = (df['Hour'] < 6) | (df['Hour'] > 18)
    time_risk = df.groupby('Is_Night')['Risk_Score'].mean()
    
    # Check if we have data for both day and night
    if True in time_risk.index and False in time_risk.index:
        time_impact = ((time_risk[True] - time_risk[False]) / time_risk[False] * 100)
        if abs(time_impact) > 5:
            risk_factors["Time: Night"] = time_impact
    elif True in time_risk.index:
        risk_factors["Time: Night"] = 100  # All crashes occurred at night
    elif False in time_risk.index:
        risk_factors["Time: Day"] = 100  # All crashes occurred during day
    
    # Analyze collision type impact
    collision_risk = df.groupby('Collision_Type')['Risk_Score'].mean()
    collision_impact = ((collision_risk - collision_risk.mean()) / collision_risk.mean() * 100)
    for collision, impact in collision_impact.items():
        if abs(impact) > 5:
            risk_factors[f"Collision Type: {collision}"] = impact
    
    return risk_factors

def analyze_crash_patterns(df):
    """
    Analyze the dataset to identify significant crash patterns.
    
    Args:
        df (pd.DataFrame): Processed traffic crash data
        
    Returns:
        list: List of identified crash patterns
    """
    patterns = []
    
    # Time-based patterns
    hourly_patterns = df.groupby('Hour')['Risk_Score'].mean()
    peak_hours = hourly_patterns[hourly_patterns > hourly_patterns.mean() + hourly_patterns.std()]
    if not peak_hours.empty:
        peak_hours_str = ", ".join([f"{hour:02d}:00" for hour in peak_hours.index])
        patterns.append(f"Peak risk hours: {peak_hours_str}")
    
    # Weather patterns
    weather_severity = df.groupby('Weather')['Severity'].value_counts(normalize=True)
    for weather in weather_severity.index.levels[0]:
        severe_ratio = weather_severity[weather].get('Severe', 0)
        if severe_ratio > 0.3:  # More than 30% severe crashes
            patterns.append(f"Higher proportion of severe crashes during {weather} conditions")
    
    # Location patterns
    location_risk = df.groupby('Location')['Risk_Score'].mean()
    high_risk_locations = location_risk[location_risk > location_risk.mean() + location_risk.std()]
    if not high_risk_locations.empty:
        patterns.append(f"Identified {len(high_risk_locations)} high-risk locations")
    
    # Collision type patterns
    collision_severity = df.groupby('Collision_Type')['Severity'].value_counts(normalize=True)
    for collision in collision_severity.index.levels[0]:
        severe_ratio = collision_severity[collision].get('Severe', 0)
        if severe_ratio > 0.3:
            patterns.append(f"Higher proportion of severe crashes in {collision} collisions")
    
    # Seasonal patterns
    df['Season'] = pd.cut(df['Date'].dt.month, 
                         bins=[0, 3, 6, 9, 12], 
                         labels=['Winter', 'Spring', 'Summer', 'Fall'])
    seasonal_risk = df.groupby('Season')['Risk_Score'].mean()
    seasonal_diff = seasonal_risk.max() - seasonal_risk.min()
    if seasonal_diff > seasonal_risk.mean() * 0.2:  # 20% difference between seasons
        high_season = seasonal_risk.idxmax()
        patterns.append(f"Higher crash risk during {high_season}")
    
    return patterns

def train_risk_prediction_model(df):
    """
    Train a machine learning model to predict crash risk based on various factors.
    
    Args:
        df (pd.DataFrame): Processed traffic crash data
        
    Returns:
        tuple: (trained model, label encoders)
    """
    # Prepare features
    features = ['Hour', 'Weather', 'Road_Condition', 'Light_Condition', 
               'Collision_Type', 'Vehicles_Involved']
    
    # Create label encoders for categorical variables
    label_encoders = {}
    X = df[features].copy()
    
    for feature in ['Weather', 'Road_Condition', 'Light_Condition', 'Collision_Type']:
        label_encoders[feature] = LabelEncoder()
        X[feature] = label_encoders[feature].fit_transform(X[feature])
    
    # Prepare target variable
    y = df['Risk_Score']
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, label_encoders

def predict_risk_for_conditions(model, label_encoders, conditions):
    """
    Predict crash risk for specific conditions using the trained model.
    
    Args:
        model: Trained RandomForestRegressor
        label_encoders (dict): Dictionary of label encoders for categorical variables
        conditions (dict): Dictionary of conditions to predict risk for
        
    Returns:
        float: Predicted risk score
    """
    # Prepare input data
    input_data = pd.DataFrame([conditions])
    
    # Encode categorical variables
    for feature, encoder in label_encoders.items():
        input_data[feature] = encoder.transform(input_data[feature])
    
    # Make prediction
    risk_score = model.predict(input_data)[0]
    
    return risk_score 