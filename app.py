import streamlit as st
import pandas as pd
import joblib
import requests
from functools import lru_cache
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# ============ CONFIG ============

API_KEY = "d3cf70bdbdf6d6ba3006a6d61d7b4ddc"  
# City meta: only lat, lon & population (industrial proximity comes from OSMnx)
city_meta = {
    "Chennai":   {"lat": 13.0827, "lon": 80.2707, "Population_Density": 0.9},
    "Coimbatore":{"lat": 11.0168, "lon": 76.9558, "Population_Density": 0.6},
    "Delhi":     {"lat": 28.7041, "lon": 77.1025, "Population_Density": 1.0},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777, "Population_Density": 1.0},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "Population_Density": 0.8},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "Population_Density": 0.8},
    "Kolkata":   {"lat": 22.5726, "lon": 88.3639, "Population_Density": 0.9},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714, "Population_Density": 0.8},
}

# Must match training feature order
FEATURE_COLS = [
    "Temperature",
    "Humidity",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Proximity_to_Industrial_Areas",
    "Population_Density",
    "roads",
    "dump_sites",
    "agricultural_fields",
]

# ============ LOAD MODEL / SCALER ============

@st.cache_resource
def load_artifacts():
    # change filenames here if needed
    model = joblib.load("xgboost_pollution_models.pkl")
    scaler = joblib.load("pollution_scalers.pkl")
    # label_encoder not required because we use manual mapping
    return model, scaler

model, scaler = load_artifacts()

# ============ OSMnx GEOSPATIAL FEATURES ============

@lru_cache(maxsize=32)
def compute_osm_features_for_city(city_name: str, radius_m: int = 3000):
    """
    Compute geospatial features around city center using OSMnx:
      - Proximity_to_Industrial_Areas (0–1)
      - roads (road segments count)
      - dump_sites (waste_disposal + landfill)
      - agricultural_fields (farmland polygons)
    Cached per city for speed.
    """
    meta = city_meta[city_name]
    lat, lon = meta["lat"], meta["lon"]

    # point in metric CRS for distance
    point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # ----- Industrial proximity -----
    try:
        industries = ox.features_from_point(
            (lat, lon),
            dist=radius_m,
            tags={"landuse": "industrial"}
        )
        if not industries.empty:
            industries = industries.to_crs(epsg=3857)
            dists = industries.geometry.distance(point.iloc[0])
            nearest_dist = dists.min()  # meters
            prox_ind = max(0.0, 1.0 - (nearest_dist / radius_m))
        else:
            prox_ind = 0.0
    except Exception:
        prox_ind = 0.0

    # ----- Roads -----
    try:
        G = ox.graph_from_point((lat, lon), dist=radius_m, network_type="drive")
        edges = ox.graph_to_gdfs(G, nodes=False)
        roads_count = len(edges)
    except Exception:
        roads_count = 0

    # ----- Dump sites -----
    dump_sites_count = 0
    try:
        dumps1 = ox.features_from_point(
            (lat, lon), dist=radius_m, tags={"amenity": "waste_disposal"}
        )
        dumps2 = ox.features_from_point(
            (lat, lon), dist=radius_m, tags={"landuse": "landfill"}
        )
        if not dumps1.empty:
            dump_sites_count += len(dumps1)
        if not dumps2.empty:
            dump_sites_count += len(dumps2)
    except Exception:
        dump_sites_count = 0

    # ----- Agricultural fields -----
    agri_count = 0
    try:
        farmland = ox.features_from_point(
            (lat, lon), dist=radius_m, tags={"landuse": "farmland"}
        )
        if not farmland.empty:
            agri_count = len(farmland)
    except Exception:
        agri_count = 0

    return {
        "Proximity_to_Industrial_Areas": float(round(prox_ind, 3)),
        "roads": int(roads_count),
        "dump_sites": int(dump_sites_count),
        "agricultural_fields": int(agri_count),
    }

# ============ OPENWEATHER FEATURES ============

def get_live_features(city_name: str):
    meta = city_meta[city_name]
    lat, lon = meta["lat"], meta["lon"]

    weather_url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    )
    air_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution?"
        f"lat={lat}&lon={lon}&appid={API_KEY}"
    )

    w = requests.get(weather_url).json()
    a = requests.get(air_url).json()

    if "main" not in w or "list" not in a:
        raise ValueError("API error – check key / quota / coordinates")

    temp = w["main"]["temp"]
    hum = w["main"]["humidity"]

    comp = a["list"][0]["components"]
    pm25 = comp.get("pm2_5")
    pm10 = comp.get("pm10")
    no2 = comp.get("no2")
    so2 = comp.get("so2")
    co = comp.get("co")

    popd = meta["Population_Density"]

    osm_feats = compute_osm_features_for_city(city_name)

    row = {
        "Temperature": temp,
        "Humidity": hum,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "Proximity_to_Industrial_Areas": osm_feats["Proximity_to_Industrial_Areas"],
        "Population_Density": popd,
        "roads": osm_feats["roads"],
        "dump_sites": osm_feats["dump_sites"],
        "agricultural_fields": osm_feats["agricultural_fields"],
    }

    df_live = pd.DataFrame([row], columns=FEATURE_COLS)
    return df_live, w, a, osm_feats

# ============ STREAMLIT UI ============

st.set_page_config(
    page_title="EnviroScan – Pollution Source Identifier",
    layout="centered"
)

st.title("🌍 EnviroScan – AI Pollution Source Identifier")
st.write(
    "Real-time prediction of pollution source using "
    "**OpenWeather air pollution + weather** and "
    "**OpenStreetMap / OSMnx geospatial features**."
)

city_choice = st.selectbox("Select City", list(city_meta.keys()), index=0)

st.write("Click the button to fetch live data and predict the pollution source.")

if st.button("🔍 Run EnviroScan Prediction"):
    try:
        # 1) Fetch live data
        X_live, w_raw, a_raw, osm_feats = get_live_features(city_choice)

        # 2) Scale features
        X_live_scaled = scaler.transform(X_live)

        # 3) Predict encoded class
        y_pred_enc = model.predict(X_live_scaled)
        code = int(y_pred_enc[0])

        # 4) Map numeric code -> human-readable label
        source_map = {
            0: "vehicle",
            1: "industry",
            2: "dust",
            3: "biomass",
            4: "mixed",
        }
        y_label_str = source_map.get(code, "unknown")

        # 5) Show live inputs
        st.subheader("Live Input Features")
        st.dataframe(X_live.style.format("{:.2f}"))

        # 6) Show geospatial context
        with st.expander("🌐 Geospatial Context (from OSMnx)"):
            st.write(f"**Industrial proximity (0–1):** {osm_feats['Proximity_to_Industrial_Areas']:.2f}")
            st.write(f"**Road segments (≈ traffic density):** {osm_feats['roads']}")
            st.write(f"**Dump sites in radius:** {osm_feats['dump_sites']}")
            st.write(f"**Agricultural fields in radius:** {osm_feats['agricultural_fields']}")

        # 7) Prediction
        st.subheader("Predicted Pollution Source")
        st.success(f"**{y_label_str.upper()}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.exception(e)
