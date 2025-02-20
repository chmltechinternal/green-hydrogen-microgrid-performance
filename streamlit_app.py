import streamlit as st

st.set_page_config(page_title="Microgrid System Analysis",
                   page_icon="🏠", layout="wide")

st.title("Microgrid System Analysis")

st.write("""
Welcome to the Green Hydrogen Microgrid Performance dashboard. This application provides insights into the performance and operation of a microgrid system, including solar PV production, load demand, hydrogen production, and fuel cell output.

Use the navigation menu on the left to explore different aspects of the analysis:
""")

st.page_link("pages/1_📊_Exploratory_Data_Analysis.py",
             label="Exploratory Data Analysis", icon="📊")
st.page_link("pages/2_📈_Hydrogen_Production_Prediction.py",
             label="Hydrogen Production Prediction", icon="📈")
st.page_link("pages/3_📈_Electrolyzer_Losses_Prediction.py",
             label="Electrolyzer Losses Prediction", icon="📈")
st.page_link("pages/4_📈_PV_Production_Prediction.py",
             label="PV Production Prediction", icon="📈")
st.page_link("pages/5_📈_System_Efficiencies_Prediction.py",
             label="System Efficiencies Prediction", icon="📈")

st.write("""
## About the Project

This dashboard analyzes data from a microgrid system, focusing on:

- Solar PV production
- Load demand patterns
- Hydrogen production and storage
- Electrolyzer losses
- System efficiencies

Explore the different pages to gain insights into the system's performance and efficiency.
""")
