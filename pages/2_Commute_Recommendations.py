import streamlit as st
from data_utils import generate_commute_recommendations

# Page config
st.set_page_config(
    page_title="Commute Recommendations - San Jose Traffic Safety",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("Commute Recommendations")
st.markdown("""
Get personalized commute recommendations based on historical traffic data and current conditions.
""")

# Check if data is available in session state
if 'filtered_df' not in st.session_state:
    st.error("No data available. Please return to the home page to load data.")
    st.stop()

# Get data from session state
filtered_df = st.session_state['filtered_df']

# Route preferences
st.sidebar.header("Route Preferences")
route_type = st.sidebar.selectbox(
    "Preferred Route Type",
    options=["Freeway", "Local Roads", "Mixed"],
    help="Select your preferred route type"
)

avoid_construction = st.sidebar.checkbox("Avoid Construction", value=True)
prefer_scenic = st.sidebar.checkbox("Prefer Scenic Routes", value=False)
avoid_high_risk = st.sidebar.checkbox("Avoid High Risk Areas", value=True)

# Generate recommendations
try:
    recommendations = generate_commute_recommendations(
        filtered_df,
        route_type=route_type,
        avoid_construction=avoid_construction,
        prefer_scenic=prefer_scenic,
        avoid_high_risk=avoid_high_risk
    )
    
    # Create columns for different types of recommendations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🕒 Time-Based Recommendations")
        time_recs = [r for r in recommendations if "🕒" in r]
        if time_recs:
            for rec in time_recs:
                st.markdown(f"- {rec}")
        else:
            st.info("No time-based recommendations available")
    
    with col2:
        st.markdown("### 🛣️ Route Recommendations")
        route_recs = [r for r in recommendations if "🛣️" in r or "⚠️" in r]
        if route_recs:
            for rec in route_recs:
                st.markdown(f"- {rec}")
        else:
            st.info("No route recommendations available")
    
    # Display weather and road condition recommendations
    st.markdown("### 🌤️ Weather & Road Conditions")
    weather_recs = [r for r in recommendations if "🌧️" in r or "🌫️" in r or "🚗" in r or "🚧" in r]
    if weather_recs:
        for rec in weather_recs:
            st.markdown(f"- {rec}")
    else:
        st.info("No weather or road condition recommendations available")
    
    # Display safety tips
    st.markdown("### 🚗 Safety Tips")
    safety_tips = [r for r in recommendations if any(emoji in r for emoji in ["🚗", "📱", "🚦", "🌙", "🔄"])]
    if safety_tips:
        for tip in safety_tips:
            st.markdown(f"- {tip}")
    else:
        st.info("No safety tips available")
        
except Exception as e:
    st.error(f"Error generating recommendations: {str(e)}") 