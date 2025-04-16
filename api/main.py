from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import httpx
import redis
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import openai
from geopy.geocoders import Nominatim
import openrouteservice as ors

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="San Jose Commute Advisor API",
    description="API for providing optimal commute recommendations for San Jose professionals",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0
)

# Initialize OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenRouteService client
ors_client = ors.Client(key=os.getenv("OPENROUTESERVICE_API_KEY"))

# Initialize geocoder
geolocator = Nominatim(user_agent="sanjose_commute_advisor")

# Models
class RoutePreference(BaseModel):
    preferred_roads: List[str]  # ["freeway", "local", "mixed"]
    start_location: str
    end_location: str
    arrival_time: str  # Format: "HH:MM"
    day_of_week: str  # Format: "Monday", "Tuesday", etc.
    weather_consideration: bool = True
    traffic_consideration: bool = True

class CommuteRecommendation(BaseModel):
    departure_time: str
    route: Dict
    estimated_duration: int
    risk_score: float
    explanation: str
    alternative_routes: List[Dict]

# Routes
@app.post("/predict", response_model=CommuteRecommendation)
async def predict_commute(preference: RoutePreference):
    """
    Predict optimal commute time and route based on user preferences.
    """
    try:
        # Generate cache key
        cache_key = f"commute:{preference.start_location}:{preference.end_location}:{preference.arrival_time}:{preference.day_of_week}"
        
        # Check cache
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Get weather data if needed
        weather_data = None
        if preference.weather_consideration:
            weather_data = await get_weather_data(preference.start_location)
        
        # Get traffic data
        traffic_data = await get_traffic_data(preference.start_location, preference.end_location)
        
        # Calculate optimal departure time
        departure_time = calculate_departure_time(
            preference.arrival_time,
            traffic_data,
            weather_data
        )
        
        # Get route recommendations
        routes = await get_route_recommendations(
            preference.start_location,
            preference.end_location,
            preference.preferred_roads
        )
        
        # Generate explanation using OpenAI
        explanation = await generate_explanation(
            routes,
            weather_data,
            traffic_data,
            departure_time
        )
        
        # Create recommendation
        recommendation = CommuteRecommendation(
            departure_time=departure_time,
            route=routes[0],  # Best route
            estimated_duration=routes[0]["duration"],
            risk_score=calculate_risk_score(routes[0], weather_data, traffic_data),
            explanation=explanation,
            alternative_routes=routes[1:]  # Alternative routes
        )
        
        # Cache the result
        redis_client.setex(
            cache_key,
            timedelta(hours=1),  # Cache for 1 hour
            json.dumps(recommendation.dict())
        )
        
        return recommendation
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/explain/{route_id}")
async def explain_route(route_id: str):
    """
    Get a detailed explanation of a specific route recommendation.
    """
    try:
        # Get route details from cache
        cache_key = f"route:{route_id}"
        route_data = redis_client.get(cache_key)
        
        if not route_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Route not found"
            )
        
        # Generate detailed explanation using OpenAI
        explanation = await generate_detailed_explanation(json.loads(route_data))
        
        return {"explanation": explanation}
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# Helper functions
async def get_weather_data(location: str) -> Dict:
    """
    Get weather data from OpenWeatherMap API.
    """
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"http://api.openweathermap.org/data/2.5/weather",
            params={
                "q": location,
                "appid": api_key,
                "units": "metric"
            }
        )
        return response.json()

async def get_traffic_data(start: str, end: str) -> Dict:
    """
    Get traffic data from OpenRouteService API.
    """
    # Get coordinates for start and end locations
    start_coords = geolocator.geocode(start)
    end_coords = geolocator.geocode(end)
    
    # Get route with traffic data
    route = ors_client.directions(
        coordinates=[[start_coords.longitude, start_coords.latitude],
                   [end_coords.longitude, end_coords.latitude]],
        profile="driving-car"
    )
    
    return route

def calculate_departure_time(
    arrival_time: str,
    traffic_data: Dict,
    weather_data: Optional[Dict]
) -> str:
    """
    Calculate optimal departure time based on traffic and weather conditions.
    """
    # Parse arrival time
    arrival = datetime.strptime(arrival_time, "%H:%M")
    
    # Calculate base travel time
    base_duration = traffic_data["features"][0]["properties"]["segments"][0]["duration"]
    
    # Adjust for weather if available
    weather_factor = 1.0
    if weather_data and weather_data.get("weather"):
        weather_condition = weather_data["weather"][0]["main"].lower()
        if weather_condition in ["rain", "snow"]:
            weather_factor = 1.5
    
    # Calculate total duration with buffer
    total_duration = base_duration * weather_factor * 1.2  # 20% buffer
    
    # Calculate departure time
    departure = arrival - timedelta(minutes=total_duration/60)
    
    return departure.strftime("%H:%M")

async def get_route_recommendations(
    start: str,
    end: str,
    preferred_roads: List[str]
) -> List[Dict]:
    """
    Get route recommendations based on user preferences.
    """
    routes = []
    
    # Get coordinates
    start_coords = geolocator.geocode(start)
    end_coords = geolocator.geocode(end)
    
    # Get routes for different preferences
    if "freeway" in preferred_roads or "mixed" in preferred_roads:
        # Get freeway route
        freeway_route = ors_client.directions(
            coordinates=[[start_coords.longitude, start_coords.latitude],
                       [end_coords.longitude, end_coords.latitude]],
            profile="driving-car",
            options={"avoid_features": ["ferry"]}
        )
        routes.append({
            "type": "freeway",
            "route": freeway_route,
            "duration": freeway_route["features"][0]["properties"]["segments"][0]["duration"]
        })
    
    if "local" in preferred_roads or "mixed" in preferred_roads:
        # Get local route
        local_route = ors_client.directions(
            coordinates=[[start_coords.longitude, start_coords.latitude],
                       [end_coords.longitude, end_coords.latitude]],
            profile="driving-car",
            options={"avoid_features": ["highway", "ferry"]}
        )
        routes.append({
            "type": "local",
            "route": local_route,
            "duration": local_route["features"][0]["properties"]["segments"][0]["duration"]
        })
    
    # Sort routes by duration
    routes.sort(key=lambda x: x["duration"])
    
    return routes

def calculate_risk_score(route: Dict, weather_data: Optional[Dict], traffic_data: Dict) -> float:
    """
    Calculate risk score for a route based on various factors.
    """
    base_risk = 0.5
    
    # Adjust for weather
    if weather_data and weather_data.get("weather"):
        weather_condition = weather_data["weather"][0]["main"].lower()
        if weather_condition in ["rain", "snow"]:
            base_risk += 0.2
    
    # Adjust for traffic
    if traffic_data.get("features"):
        congestion = traffic_data["features"][0]["properties"].get("congestion", 0)
        base_risk += congestion * 0.1
    
    # Adjust for route type
    if route["type"] == "freeway":
        base_risk += 0.1  # Freeways generally have higher risk
    
    return min(base_risk, 1.0)

async def generate_explanation(
    routes: List[Dict],
    weather_data: Optional[Dict],
    traffic_data: Dict,
    departure_time: str
) -> str:
    """
    Generate natural language explanation using OpenAI API.
    """
    prompt = f"""
    Generate a concise explanation for a commute recommendation with the following details:
    - Departure time: {departure_time}
    - Best route type: {routes[0]['type']}
    - Estimated duration: {routes[0]['duration']/60:.1f} minutes
    - Weather conditions: {weather_data['weather'][0]['description'] if weather_data else 'Unknown'}
    - Traffic conditions: {'Congested' if traffic_data.get('features', [{}])[0].get('properties', {}).get('congestion', 0) > 0.5 else 'Normal'}
    
    Focus on the key factors that influenced the recommendation and provide practical advice for the commute.
    """
    
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful commute advisor providing clear, concise explanations."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

async def generate_detailed_explanation(route_data: Dict) -> str:
    """
    Generate a detailed explanation of a specific route using OpenAI API.
    """
    prompt = f"""
    Provide a detailed analysis of this commute route:
    {json.dumps(route_data, indent=2)}
    
    Include:
    1. Route overview and key landmarks
    2. Potential traffic hotspots
    3. Safety considerations
    4. Alternative options
    5. Tips for a smooth commute
    """
    
    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a detailed commute advisor providing comprehensive route analysis."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content 