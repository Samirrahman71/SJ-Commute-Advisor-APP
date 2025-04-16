import streamlit as st
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap, MarkerCluster
from data_utils import format_time_12hr

# Page config
st.set_page_config(
    page_title="Incident Map - San Jose Traffic Safety",
    page_icon="🗺️",
    layout="wide"
)

# Title
st.title("Incident Map")
st.markdown("""
This interactive map shows traffic incidents in San Jose, CA.
Use the filters to customize your view.
""")

# Check if data is available in session state
if 'filtered_df' not in st.session_state:
    st.error("No data available. Please return to the home page to load data.")
    st.stop()

# Get data from session state
filtered_df = st.session_state['filtered_df']

# Map visualization options
st.sidebar.header("Map Options")
show_heatmap = st.sidebar.checkbox("Show Heatmap", value=True)
show_markers = st.sidebar.checkbox("Show Individual Incidents", value=True)
cluster_markers = st.sidebar.checkbox("Cluster Markers", value=True)

# Create map
m = folium.Map(location=[37.3382, -121.8863], zoom_start=12)

# Add heatmap if enabled
if show_heatmap and not filtered_df.empty:
    try:
        heatmap_data = filtered_df[['Latitude', 'Longitude', 'Risk_Score']].values.tolist()
        HeatMap(heatmap_data, radius=15, blur=10).add_to(m)
    except Exception as e:
        st.warning(f"Could not create heatmap: {str(e)}")

# Create marker cluster if enabled
if cluster_markers:
    marker_cluster = MarkerCluster().add_to(m)

# Add markers for each incident
if show_markers:
    for idx, row in filtered_df.iterrows():
        try:
            # Create popup content with more details
            popup_content = f"""
            <div style="font-family: Arial, sans-serif;">
                <h4 style="margin-bottom: 5px;">Incident Details</h4>
                <p><b>Date:</b> {row['Date'].strftime('%Y-%m-%d')}</p>
                <p><b>Time:</b> {format_time_12hr(row['Hour'])}</p>
                <p><b>Location:</b> {row['Location']}</p>
                <p><b>Severity:</b> {row['Severity']}</p>
                <p><b>Risk Score:</b> {row['Risk_Score']:.2f}</p>
                <p><b>Weather:</b> {row['Weather']}</p>
                <p><b>Road Condition:</b> {row['Road_Condition']}</p>
                <p><b>Movement:</b> {row['MovementPrecedingCollision']}</p>
            </div>
            """
            
            # Determine marker color based on risk score
            if row['Risk_Score'] > 0.7:
                color = 'darkred'
            elif row['Risk_Score'] > 0.4:
                color = 'orange'
            else:
                color = 'lightgreen'
            
            # Create marker
            marker = folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_content, max_width=300),
                icon=folium.Icon(color=color, icon='info-sign'),
                tooltip=f"Risk Score: {row['Risk_Score']:.2f}"
            )
            
            # Add to cluster or map
            if cluster_markers:
                marker.add_to(marker_cluster)
            else:
                marker.add_to(m)
        except Exception as e:
            st.warning(f"Could not add marker for incident {idx}: {str(e)}")
            continue

# Display the map
st.write("Use the map below to explore traffic incidents. Click on markers for details.")
folium_static(m, width=1200, height=600)

# Display map legend
st.markdown("""
### Map Legend
- **Red Markers**: High risk incidents (Risk Score > 0.7)
- **Orange Markers**: Medium risk incidents (Risk Score 0.4 - 0.7)
- **Green Markers**: Low risk incidents (Risk Score < 0.4)
""") 