import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import math
import filters

st.set_page_config(page_title="Electrolyzer Losses Prediction", layout="wide")

st.title("Electrolyzer Losses Prediction")
st.write("""
This application predicts electrolyzer losses for a microgrid system based on Hydrogen production and Energy supplied to Electrolyzer (E_PV2EZ).
Choose a model, adjust parameters, and explore the results!
""")

# Load and display data
df = filters.load_data()

# Dataset overview
st.header("Dataset Overview")
st.write("Here's a glimpse of the data we're working with:")
# st.dataframe(df.head())
st.dataframe(df[['DateTime', 'Month', 'Hour', 'Electrolyzer Losses (kWh)', 'E_PV2EZ (kWh)', 'Hydrogen Production (kg)']].head())