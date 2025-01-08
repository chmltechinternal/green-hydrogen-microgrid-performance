import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page setup
st.set_page_config(page_title="Microgrid Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
   df = pd.read_csv('data/Full Dataset with months.csv')
   df['DateTime'] = pd.to_datetime(df['DateTime'])
   df['Month'] = df['DateTime'].dt.strftime('%B')
   df['Hour'] = df['DateTime'].dt.hour
   return df

# Load the data
df = load_data()

# Sidebar filters
st.sidebar.header('Filters')

# Month selection
selected_month = st.sidebar.selectbox(
   'Select Month',
   options=df['Month'].unique()
)

# Time range selection within day
hour_range = st.sidebar.slider(
   'Select Hour Range',
   min_value=0,
   max_value=23,
   value=(0, 23)
)

# Filter data
filtered_df = df[
   (df['Month'] == selected_month) &
   (df['Hour'].between(hour_range[0], hour_range[1]))
]

# Main dashboard
st.title('Microgrid System Analysis Dashboard')

# Top row metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
   avg_pv = filtered_df['PV Production (kWh)'].mean()
   st.metric("Avg PV Production", f"{avg_pv:.2f} kWh")

with col2:
   avg_load = filtered_df['Load Demand (kWh)'].mean()
   st.metric("Avg Load Demand", f"{avg_load:.2f} kWh")

with col3:
   avg_h2_prod = filtered_df['Hydrogen Production (kg)'].mean()
   st.metric("Avg H₂ Production", f"{avg_h2_prod:.2f} kg")

with col4:
   avg_fc_output = filtered_df['Fuel Cell Output (kWh)'].mean()
   st.metric("Avg Fuel Cell Output", f"{avg_fc_output:.2f} kWh")

# Energy Balance Chart
st.subheader('Energy Balance Overview')
energy_fig = go.Figure()

energy_fig.add_trace(go.Scatter(
   x=filtered_df['DateTime'],
   y=filtered_df['PV Production (kWh)'],
   name='PV Production',
   line=dict(color='#FFA500')
))

energy_fig.add_trace(go.Scatter(
   x=filtered_df['DateTime'],
   y=filtered_df['Load Demand (kWh)'],
   name='Load Demand',
   line=dict(color='#FF0000')
))

energy_fig.add_trace(go.Scatter(
   x=filtered_df['DateTime'],
   y=filtered_df['Energy Imported from Grid (kWh)'],
   name='Grid Import',
   line=dict(color='#00FF00')
))

energy_fig.update_layout(
   xaxis_title='Date',
   yaxis_title='Energy (kWh)',
   hovermode='x unified'
)

st.plotly_chart(energy_fig, use_container_width=True)

# Hydrogen System Charts
col1, col2 = st.columns(2)

with col1:
   h2_fig = go.Figure()
   h2_fig.add_trace(go.Scatter(
       x=filtered_df['DateTime'],
       y=filtered_df['Hydrogen Production (kg)'],
       name='H₂ Production',
       line=dict(color='#0000FF')
   ))
   h2_fig.add_trace(go.Scatter(
       x=filtered_df['DateTime'],
       y=filtered_df['Hydrogen Supply to Fuel Cell (kg)'],
       name='H₂ to Fuel Cell',
       line=dict(color='#800080')
   ))
   h2_fig.update_layout(
       title='Hydrogen Production and Usage',
       xaxis_title='Date',
       yaxis_title='Hydrogen (kg)',
       hovermode='x unified'
   )
   st.plotly_chart(h2_fig, use_container_width=True)

with col2:
   eff_fig = go.Figure()
   eff_fig.add_trace(go.Scatter(
       x=filtered_df['DateTime'],
       y=filtered_df['Electrolyser Efficiency (%)'],
       name='Electrolyzer Eff.',
       line=dict(color='#4B0082')
   ))
   eff_fig.add_trace(go.Scatter(
       x=filtered_df['DateTime'],
       y=filtered_df['Fuel Cell Electrical Efficiency (%)'],
       name='Fuel Cell Eff.',
       line=dict(color='#008080')
   ))
   eff_fig.update_layout(
       title='System Efficiencies',
       xaxis_title='Date',
       yaxis_title='Efficiency (%)',
       hovermode='x unified'
   )
   st.plotly_chart(eff_fig, use_container_width=True)

# Daily Patterns section
st.subheader('Daily Operating Patterns')

# Calculate daily averages
daily_avg = filtered_df.groupby(filtered_df['DateTime'].dt.hour).agg({
   'PV Production (kWh)': 'mean',
   'Load Demand (kWh)': 'mean',
   'Hydrogen Production (kg)': 'mean',
   'Fuel Cell Output (kWh)': 'mean'
}).reset_index()

# Daily patterns visualization
daily_fig = go.Figure()
daily_fig.add_trace(go.Scatter(
   x=daily_avg['DateTime'],
   y=daily_avg['PV Production (kWh)'],
   name='Avg PV Production',
   line=dict(color='#FFA500')
))
daily_fig.add_trace(go.Scatter(
   x=daily_avg['DateTime'],
   y=daily_avg['Load Demand (kWh)'],
   name='Avg Load Demand',
   line=dict(color='#FF0000')
))

daily_fig.update_layout(
   title='24-Hour Energy Profile',
   xaxis_title='Hour of Day',
   yaxis_title='Energy (kWh)',
   hovermode='x unified'
)

st.plotly_chart(daily_fig, use_container_width=True)

# System Summary Metrics
st.subheader('System Performance Summary')
col1, col2, col3 = st.columns(3)

with col1:
   avg_export = filtered_df['Energy Exported to Grid (kWh)'].mean()
   avg_import = filtered_df['Energy Imported from Grid (kWh)'].mean()
   st.metric("Grid Export vs Import", 
             f"Export: {avg_export:.1f} kWh",
             f"Import: {avg_import:.1f} kWh")

with col2:
   avg_h2_storage = filtered_df['Hydrogen Discharge to Storage (kg)'].mean()
   st.metric("Avg H₂ Storage", f"{avg_h2_storage:.2f} kg")

with col3:
   avg_sys_eff = filtered_df['Electrolyser Efficiency (%)'].mean()
   st.metric("Avg System Efficiency", f"{avg_sys_eff:.1f}%")