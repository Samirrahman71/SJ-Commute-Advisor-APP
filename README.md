# San Jose Commute Advisor

A comprehensive commute recommendation system for San Jose professionals, helping them plan optimal routes and departure times to arrive at work by 9 AM.

## Features

- **Personalized Route Recommendations**: Choose between freeways, local roads, or a combination
- **Real-time Traffic Analysis**: Consider current traffic conditions and historical patterns
- **Weather Integration**: Account for weather conditions that might affect commute times
- **AI-Powered Insights**: Get natural language explanations of recommendations
- **Interactive Maps**: Visualize recommended routes and alternatives
- **Historical Analysis**: View crash data and risk factors by time, location, and conditions

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- API keys for:
  - OpenAI
  - OpenWeatherMap
  - OpenRouteService

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sanjose-commute-advisor.git
   cd sanjose-commute-advisor
   ```

2. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
   OPENROUTESERVICE_API_KEY=your_openrouteservice_api_key
   ```

3. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. Access the application:
   - Streamlit UI: http://localhost:8501
   - FastAPI docs: http://localhost:8000/docs

## API Endpoints

### POST /predict
Get commute recommendations based on preferences.

Request body:
```json
{
  "preferred_roads": ["freeway", "local", "mixed"],
  "start_location": "Your home address",
  "end_location": "Work address",
  "arrival_time": "09:00",
  "day_of_week": "Monday",
  "weather_consideration": true,
  "traffic_consideration": true
}
```

### GET /explain/{route_id}
Get detailed explanation of a specific route.

## Development

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   # Terminal 1: FastAPI
   uvicorn api.main:app --reload
   
   # Terminal 2: Streamlit
   streamlit run app.py
   ```

## Testing

Run tests with pytest:
```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- California Department of Transportation for traffic data
- OpenStreetMap for map data
- OpenAI for AI-powered insights
- OpenWeatherMap for weather data
- OpenRouteService for routing data 