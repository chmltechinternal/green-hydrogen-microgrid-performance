import bokeh
import streamlit as st
import pandas as pd
# import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import filters

st.set_page_config(page_title="Hydrogen Production Prediction", layout="wide")

st.title("Hydrogen Production Prediction")

st.write("This page is under construction. It will contain Hydrogen Production Prediction for the microgrid system.")

df = filters.load_data()

# Display dataset overview
st.subheader("Dataset Overview")
st.dataframe(df.head())

# Feature and target columns
feature_col = ['Solar Irradiance (W/m^2)', 'Ambient Temperature (°C)']
output = ['Hydrogen Production (kg)']

data = df[['DateTime', 'Month', 'Hour'] + feature_col + output]

# Model selection
model_type = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Prophet"])

def get_list_nnan(column_name):
  list_y_nan = list(df[column_name][df[column_name].apply(lambda x: math.isnan(x))].index)
  list_x_nan =[]
  for i in feature_col:
    list_x_nan+= list(df[i][df[i].apply(lambda x: math.isnan(x))].index)
  list_x_nan =list(set(list_x_nan))
  list_nan = list(set(list_x_nan+list_y_nan))
  list_notnan = [i for i in list(df.index) if i not in list_nan]
  return list_notnan

list_notnan  = get_list_nnan('Hydrogen Production (kg)')
X=df[feature_col].loc[list_notnan]
y= df['Hydrogen Production (kg)'].loc[list_notnan]

# Model training and prediction
if model_type == "Random Forest":
    st.subheader("Random Forest Regressor")
    n_estimators = st.slider("Number of Estimators", 10, 100, 10)
    max_depth = st.slider("Max Depth", 5, 20, 10)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
    )
    regr = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=13)
    regr.fit(X_train, y_train)
    
    # Prediction
    y_pred = regr.predict(X_test) 
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    st.subheader("Enter New Data for Prediction")
    solar_irradiance = st.number_input("Solar Irradiance (W/m^2)", value=500)
    ambient_temperature = st.number_input("Ambient Temperature (°C)", value=25)

    new_data = pd.DataFrame([[solar_irradiance, ambient_temperature]], columns=feature_col)
    prediction = regr.predict(new_data.values.reshape(1, -1))  # or reg.predict(new_data)
    st.write(f"Predicted Hydrogen Production: {prediction[0]} kg")
    # Evaluation
    

elif model_type == "Gradient Boosting":
    st.subheader("Gradient Boosting Regressor")
    n_estimators = st.slider("Number of Estimators", 10, 100, 10)
    max_depth = st.slider("Max Depth", 5, 20, 10)
    
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=13
    )
    regr = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=13)
    regr.fit(X_train, y_train)
    
    y_pred = regr.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse}")

    st.subheader("Enter New Data for Prediction")
    solar_irradiance = st.number_input("Solar Irradiance (W/m^2)", value=500)
    ambient_temperature = st.number_input("Ambient Temperature (°C)", value=25)

    new_data = pd.DataFrame([[solar_irradiance, ambient_temperature]], columns=feature_col)
    prediction = regr.predict(new_data.values.reshape(1, -1))  # or reg.predict(new_data)
    st.write(f"###### Predicted Hydrogen Production: *_{prediction[0]} kg_*")

elif model_type == "Prophet":
    st.subheader("Prophet")

    train_data = data.iloc[:7000]
    test_data = data.iloc[7000:]

    model= Prophet()
    # model.fit(train_data[['ds','y']])
    # model = Prophet(growth='logistic')
    # Add the external regressor(s) to the model
    model.add_regressor('Solar Irradiance (W/m^2)')
    model.add_regressor('Ambient Temperature (°C)')
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.fit(train_data)

    test_p = data[['ds','Solar Irradiance (W/m^2)', 'Ambient Temperature (°C)']]
    forecast = model.predict(test_p)
    
    # (Evaluation)

# Visualization
st.subheader("Visualization")

dataframes = [df, data]
filters.initialize_filters(dataframes)
filters.display_filters()
filtered_dataframes = [filters.apply_filters(df) for df in dataframes]

for i, df in enumerate(filtered_dataframes):
    if i == 0:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['DateTime'],
            y=df['Hydrogen Production (kg)'],
            mode='lines',
            name='Hydrogen Production',
            line=dict(color='#0f2113')
        ))

        fig.update_layout(
            autosize=True,
            margin=dict(l=40, r=20, t=60, b=40),
            title={'text': f'Hydrogen Production for {st.session_state.selected_month} (Hours: {st.session_state.hour_range[0]}-{st.session_state.hour_range[1]})', 'x': 0.5,
            'xanchor': 'center', 'font': dict(color='#0f2113')},
            xaxis_title='Date',
            yaxis_title='Hydrogen Production (kg)',
            plot_bgcolor='#ffffff',
            paper_bgcolor='#4bd659',
            font=dict(color='#0f2113'),
            hovermode='x unified'
        )

        fig.update_xaxes(
            tickformat='%Y-%m-%d %H:%M',
            tickfont=dict(color='#0f2113'),
            titlefont=dict(color='#0f2113')
        )

        fig.update_yaxes(
            tickfont=dict(color='#0f2113'),
            titlefont=dict(color='#0f2113')
        )

        st.plotly_chart(fig, use_container_width=True)

# ... (Other visualizations like Feature Importance and Prediction vs. Actual)
