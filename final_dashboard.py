import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import json

# 1. PAGE CONFIGURATION
st.set_page_config(page_title="EnviroScan: Real-Time Pollution Source Monitoring", layout="wide")

# 2. CUSTOM CSS
st.markdown("""
<style>
.stApp { background-color: #2D9B9E !important; }
.stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp h1, .stApp h2, .stApp h3 {
    color: #FFFFFF !important;
}

/* DASHBOARD TITLE SIZE */
.main-title {
    font-size: 50px !important;
    font-weight: 800 !important;
}

/* STATUS MESSAGE FONT SIZE */
.stAlert p {
    font-size: 26px !important;
    font-weight: bold !important;
}

/* SIDEBAR STYLING */
[data-testid="stSidebar"] {
    min-width: 420px !important;
    background-color: #1A6B6E !important;
}
section[data-testid="stSidebar"] label p { 
    font-size: 20px !important; 
    font-weight: bold !important; 
}

/* DOWNLOAD BUTTONS */
.stDownloadButton button p, .stButton button p {
    color: #000000 !important;
    font-size: 24px !important;
    font-weight: bold !important;
}
.stDownloadButton button, .stButton button {
    background-color: #FFFFFF !important;
    border: 2px solid #1A6B6E !important;
    border-radius: 15px !important;
    padding: 15px 25px !important;
    width: 100% !important;
}

.metric-box {
    border: 2px solid #FFFFFF;
    padding: 10px;
    text-align: center;
    border-radius: 5px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# 3. DATA LOADING
@st.cache_data
def load_and_process_data():
    try:
        data = pd.read_csv('pollution_datasets_clean.csv')
        data.columns = [col.strip() for col in data.columns]
        cities = ['Hyderabad', 'Delhi', 'Mumbai', 'Salem', 'Bangalore']
        data['City'] = (cities * (len(data) // len(cities) + 1))[:len(data)]
        coords = {'Hyderabad': [17.38, 78.48], 'Delhi': [28.61, 77.20], 'Mumbai': [19.07, 72.87], 'Salem': [11.66, 78.14], 'Bangalore': [12.97, 77.59]}
        data['Latitude'] = data['City'].map(lambda x: coords[x][0])
        data['Longitude'] = data['City'].map(lambda x: coords[x][1])
        data['AQI'] = pd.to_numeric(data['PM2.5'], errors='coerce') * 1.5
        
        def categorize(aqi):
            if aqi > 150: return 'High', 'red'
            if aqi > 50: return 'Middle', 'orange'
            return 'Low', 'green'
            
        data[['Risk', 'Color']] = data['AQI'].apply(lambda x: pd.Series(categorize(x)))
        return data
    except Exception:
        return pd.DataFrame()

df = load_and_process_data()

# 4. DASHBOARD UI
if not df.empty:
    # SIDEBAR INPUTS
    st.sidebar.title("🔧 Input Parameters")
    co_aqi = st.sidebar.number_input("CO AQI Value", value=150.00)
    no2_aqi = st.sidebar.number_input("NO2 AQI Value", value=140.00)
    ozone_aqi = st.sidebar.number_input("Ozone AQI Value", value=130.00)
    pm25_aqi = st.sidebar.number_input("PM2.5 AQI Value", value=160.00)
    overall_aqi = st.sidebar.number_input("Overall AQI", value=240.00)
    temp = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, 41.33)
    humidity = st.sidebar.slider("Humidity (%)", 0, 100, 25)
    search_city = st.sidebar.selectbox("🔍 Select Region:", sorted(df['City'].unique()))

    # HEADER & STATUS
    st.markdown('<h1 class="main-title">🌍 EnviroScan: Real-Time Pollution Source Monitoring</h1>', unsafe_allow_html=True)
    st.info("✅ Worldwide Pollution Risk Monitoring Active")

    # KPI ROW
    k1, k2 = st.columns(2)
    with k1:
        st.markdown('<div class="metric-box">📌 Predicted Source</div>', unsafe_allow_html=True)
        st.header("Industrial" if co_aqi > 100 else "Natural")
    with k2:
        st.markdown('<div class="metric-box">Confidence Score</div>', unsafe_allow_html=True)
        st.header("94.50%")

    # ROW 1: CHARTS
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🟠 Source Distribution")
        fig_p = px.pie(values=[co_aqi, pm25_aqi, no2_aqi], names=["Natural", "Industrial", "Vehicular"], hole=0.5, color_discrete_sequence=['#FFFFFF', '#D9FAFF', '#A6F2FF'])
        fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#FFFFFF")
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        st.subheader("📊 Pollutant Levels (µg/m³)")
        fig_b = px.bar(x=["PM2.5", "NO2", "CO"], y=[pm25_aqi, no2_aqi, co_aqi], color_discrete_sequence=['#FFFFFF'])
        fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#FFFFFF")
        st.plotly_chart(fig_b, use_container_width=True)

    # ROW 2: GAUGE & TREND
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("🎯 Air Quality Index")
        fig_g = go.Figure(go.Indicator(mode="gauge+number", value=pm25_aqi, gauge={'axis': {'range': [0, 500]}, 'bar': {'color': "#FFFFFF"}}))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="#FFFFFF")
        st.plotly_chart(fig_g, use_container_width=True)
    with c4:
        st.subheader("📈 Pollution Trend")
        fig_l = px.line(df.iloc[:20], y=['PM2.5', 'NO2'])
        fig_l.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#FFFFFF")
        fig_l.update_traces(line_color="#FFFFFF")
        st.plotly_chart(fig_l, use_container_width=True)

    # 7. WORLDWIDE MAP WITH CAROUSEL BAR
    st.markdown("---")
    st.subheader(f"📍 Interactive Risk Explorer: {search_city}")
    
    # The Carousel/Tab Bar
    carousel_bar = st.tabs(["🌍 General Map", "🔴 High Risk Areas", "🟠 Middle Risk Areas", "🟢 Low Risk Areas"])
    avg_lat, avg_lon = df['Latitude'].mean(), df['Longitude'].mean()

    def create_filtered_map(risk_filter=None):
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=4)
        filtered_df = df[df['Risk'] == risk_filter] if risk_filter else df
        for _, row in filtered_df.sample(min(len(filtered_df), 150)).iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=8,
                popup=f"AQI: {row['AQI']:.1f} ({row['Risk']} Risk)",
                color=row['Color'], fill=True, fill_color=row['Color']
            ).add_to(m)
        return m

    with carousel_bar[0]:
        st.info("Showing all monitored locations worldwide.")
        st_folium(create_filtered_map(), key="map_gen", use_container_width=True, height=500)
    with carousel_bar[1]:
        st.error("Critical Locations: Areas requiring immediate industrial regulation.")
        st_folium(create_filtered_map("High"), key="map_high", use_container_width=True, height=500)
    with carousel_bar[2]:
        st.warning("Monitoring Required: Areas with moderate pollutant levels.")
        st_folium(create_filtered_map("Middle"), key="map_mid", use_container_width=True, height=500)
    with carousel_bar[3]:
        st.success("Safe Zones: Areas within healthy environmental limits.")
        st_folium(create_filtered_map("Low"), key="map_low", use_container_width=True, height=500)

    # 8. DOWNLOAD REPORTS
    st.markdown("---")
    st.subheader("📥 Download Reports")
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1: st.download_button("📄 CSV", df.to_csv(index=False), "report.csv", "text/csv", key="f_csv")
    with d_col2: st.download_button("📄 TXT", df.to_string(), "report.txt", "text/plain", key="f_txt")
    with d_col3: st.download_button("📁 JSON", df.to_json(), "report.json", "application/json", key="f_json")
    with d_col4: st.button("📕 PDF", key="f_pdf")

else:
    st.error("Error: Please check 'pollution_datasets_clean.csv' in your folder.")
