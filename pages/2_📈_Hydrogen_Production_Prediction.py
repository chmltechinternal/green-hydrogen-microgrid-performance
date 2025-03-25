import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import plotly.express as px
import math
import filters

st.set_page_config(page_title="Hydrogen Production Prediction", layout="wide")

st.title("Hydrogen Production Prediction")
st.write("""
This application predicts hydrogen production for a microgrid system based on solar irradiance and ambient temperature.
Choose a model, adjust parameters, and explore the results!
""")

df = filters.load_data()

# Dataset overview
st.header("Dataset Overview")
st.write("Here's a glimpse of the data we're working with:")
st.dataframe(df[['DateTime', 'Month', 'Hour', 'Solar Irradiance (W/m^2)',
             'Ambient Temperature (°C)', 'Hydrogen Production (kg)']].head())

feature_col = ['Solar Irradiance (W/m^2)', 'Ambient Temperature (°C)']
output = ['Hydrogen Production (kg)']

data = df[['DateTime', 'Month', 'Hour'] + feature_col + output]


st.subheader("Compare Different  Model Predictions")
tab_titles = ["Random Forest", "Gradient Boosting", "Prophet"]
tabs = st.tabs(tab_titles)


def get_list_nnan(column_name):
    list_y_nan = list(
        df[column_name][df[column_name].apply(lambda x: math.isnan(x))].index)
    list_x_nan = []
    for i in feature_col:
        list_x_nan += list(df[i][df[i].apply(lambda x: math.isnan(x))].index)
    list_x_nan = list(set(list_x_nan))
    list_nan = list(set(list_x_nan + list_y_nan))
    list_notnan = [i for i in list(df.index) if i not in list_nan]
    return list_notnan


list_notnan = get_list_nnan('Hydrogen Production (kg)')
X = df[feature_col].loc[list_notnan]
y = df['Hydrogen Production (kg)'].loc[list_notnan]


with tabs[0]:
    st.write(f"### Random Forest Model Training and Prediction")
    colA, colB = st.columns([2, 4], gap="large", vertical_alignment="center")

    with colA:
        st.subheader("Train Model")
        st.write("Select the number of estimators and max depth to train the model and view different metrics to help determine the suitability of the model for hydrogen production prediction")
        n_estimators = st.slider(
            "Number of Estimators", 10, 200, 50, help="The number of trees in the forest.", key="rf_n_est")
        max_depth = st.slider("Max Depth", 5, 20, 10,
                              help="The maximum depth of the tree.", key="rf_max_dpth")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        regr = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        regr.fit(X_train, y_train)
        y_pred_train = regr.predict(X_train)
        y_pred_test = regr.predict(X_test)

        df_results = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred_test
        })

    with colB:
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        col1, col2, col3, col4 = st.columns(4, border=True)
        col5, col6, col7, col8 = st.columns(4, border=True)

        def create_metric_card(container, title, value):
            with container:
                st.markdown(
                    f"<div><h6 style='text-align: center;'>{title}</h6></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<h6 style='text-align: center;'>{value:.2f}</h6>", unsafe_allow_html=True)

        # Create cards for each metric
        create_metric_card(col1, "MSE (Train)", mse_train)
        create_metric_card(col2, "R² (Train)", r2_train)
        create_metric_card(col3, "RMSE (Train)", rmse_train)
        create_metric_card(col4, "MAE (Train)", mae_train)
        create_metric_card(col5, "MSE (Test)", mse_test)
        create_metric_card(col6, "R² (Test)", r2_test)
        create_metric_card(col7, "RMSE (Test)", rmse_test)
        create_metric_card(col8, "MAE (Test)", mae_test)

    col9, col10 = st.columns(2, gap="large", vertical_alignment="center")

    with col9:
        ''
        ''
        st.subheader("Predict Hydrogen Production")
        st.write(
            "Enter new values for solar irradiance and ambient temperature to predict hydrogen production:")
        solar_irradiance = st.number_input(
            "Solar Irradiance (W/m^2)", value=500, step=10, key="rf_solar_irr")
        ambient_temperature = st.number_input(
            "Ambient Temperature (°C)", value=25, step=1, key="rf_amb__temp")

        new_data = pd.DataFrame(
            [[solar_irradiance, ambient_temperature]], columns=feature_col)
        prediction = regr.predict(new_data)[0]

    with col10:
        st.success(f"Predicted Hydrogen Production: {prediction:.2f} kg")

    ''
    ''
    st.subheader("Predicted vs Actual")
    st.write("""
    The predicted vs actual chart compares the model's predictions with the actual observed values.
    Points closer to the diagonal line indicate more accurate predictions.
    Deviations from this line show where the model over- or under-predicts.
    This visualization helps assess the model's overall performance and identify any systematic prediction errors.
    """)

    train_col, test_col = st.columns(2)

    with train_col:
        fig_pred_actual = px.scatter(
            x=y_train,
            y=y_pred_train,
            labels={'x': 'Actual Values (Train)',
                    'y': 'Predicted Values (Train)'},
            title='Training Set: Actual vs Predicted',
            trendline='ols',
            trendline_color_override='red'
        )
        fig_pred_actual.add_shape(
            type="line", line=dict(dash='dash'),
            x0=y_train.min(), y0=y_train.min(),
            x1=y_train.max(), y1=y_train.max()
        )

        st.plotly_chart(fig_pred_actual, use_container_width=True,
                        key='rf_actual_pred_train')

    with test_col:
        fig_test = px.scatter(
            x=y_test,
            y=y_pred_test,
            labels={'x': 'Actual Values (Test)',
                    'y': 'Predicted Values (Test)'},
            title='Test Set: Actual vs Predicted',
            trendline='ols',
            trendline_color_override='red'
        )
        fig_test.add_shape(
            type="line", line=dict(dash='dash'),
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max()
        )
        st.plotly_chart(fig_test, key='rf_actual_pred_test')

    st.subheader("Feature Importance")
    st.write("""
    Feature importance shows the relative importance of each input variable in predicting the target variable.
    Higher values indicate that the feature has a stronger influence on the model's predictions.
    This can help identify which factors are most crucial for hydrogen production in your system.
    """)
    importance = regr.feature_importances_
    feature_importance = pd.DataFrame(
        {'feature': feature_col, 'importance': importance})
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False)

    fig_importance = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    fig_importance.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=400
    )
    st.plotly_chart(fig_importance, use_container_width=True, key='rf_imp')

with tabs[1]:
    st.write(f"### Gradient Boosting Model Training and Prediction")
    colA, colB = st.columns([2, 4], gap="large", vertical_alignment="center")

    with colA:
        st.subheader("Train Model")
        st.write("Select the number of estimators and max depth to train the model and view different metrics to help determine the suitability of the model for hydrogen production prediction")
        n_estimators = st.slider(
            "Number of Estimators", 10, 200, 50, help="The number of trees in the forest.", key="gb_n_est")
        max_depth = st.slider("Max Depth", 5, 20, 10,
                              help="The maximum depth of the tree.", key="gb_max_dpth")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        regr = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)

        regr.fit(X_train, y_train)
        y_pred_train = regr.predict(X_train)
        y_pred_test = regr.predict(X_test)

        df_results = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred_test
        })

    with colB:
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)

        col1, col2, col3, col4 = st.columns(4, border=True)
        col5, col6, col7, col8 = st.columns(4, border=True)

        def create_metric_card(container, title, value):
            with container:
                st.markdown(
                    f"<div><h6 style='text-align: center;'>{title}</h6></div>", unsafe_allow_html=True)
                st.markdown(
                    f"<h6 style='text-align: center;'>{value:.2f}</h6>", unsafe_allow_html=True)

        # Create cards for each metric
        create_metric_card(col1, "MSE (Train)", mse_train)
        create_metric_card(col2, "R² (Train)", r2_train)
        create_metric_card(col3, "RMSE (Train)", rmse_train)
        create_metric_card(col4, "MAE (Train)", mae_train)
        create_metric_card(col5, "MSE (Test)", mse_test)
        create_metric_card(col6, "R² (Test)", r2_test)
        create_metric_card(col7, "RMSE (Test)", rmse_test)
        create_metric_card(col8, "MAE (Test)", mae_test)

    col9, col10 = st.columns(2, gap="large", vertical_alignment="center")

    with col9:
        ''
        ''
        st.subheader("Predict Hydrogen Production")
        st.write(
            "Enter new values for solar irradiance and ambient temperature to predict hydrogen production:")
        solar_irradiance = st.number_input(
            "Solar Irradiance (W/m^2)", value=500, step=10, key="gb_solar_irr")
        ambient_temperature = st.number_input(
            "Ambient Temperature (°C)", value=25, step=1, key="gb_amb_temp")

        new_data = pd.DataFrame(
            [[solar_irradiance, ambient_temperature]], columns=feature_col)
        prediction = regr.predict(new_data)[0]

    with col10:
        st.success(f"Predicted Hydrogen Production: {prediction:.2f} kg")

    ''
    ''
    st.subheader("Predicted vs Actual")
    st.write("""
    The predicted vs actual chart compares the model's predictions with the actual observed values.
    Points closer to the diagonal line indicate more accurate predictions.
    Deviations from this line show where the model over- or under-predicts.
    This visualization helps assess the model's overall performance and identify any systematic prediction errors.
    """)

    train_col, test_col = st.columns(2)

    with train_col:
        fig_pred_actual = px.scatter(
            x=y_train,
            y=y_pred_train,
            labels={'x': 'Actual Values (Train)',
                    'y': 'Predicted Values (Train)'},
            title='Training Set: Actual vs Predicted',
            trendline='ols',
            trendline_color_override='red'
        )
        fig_pred_actual.add_shape(
            type="line", line=dict(dash='dash'),
            x0=y_train.min(), y0=y_train.min(),
            x1=y_train.max(), y1=y_train.max()
        )

        st.plotly_chart(fig_pred_actual, use_container_width=True,
                        key='gb_actual_pred_train')

    with test_col:
        fig_test = px.scatter(
            x=y_test,
            y=y_pred_test,
            labels={'x': 'Actual Values (Test)',
                    'y': 'Predicted Values (Test)'},
            title='Test Set: Actual vs Predicted',
            trendline='ols',
            trendline_color_override='red'
        )
        fig_test.add_shape(
            type="line", line=dict(dash='dash'),
            x0=y_test.min(), y0=y_test.min(),
            x1=y_test.max(), y1=y_test.max()
        )
        st.plotly_chart(fig_test, key='gb_actual_pred_test')

    st.subheader("Feature Importance")
    st.write("""
    Feature importance shows the relative importance of each input variable in predicting the target variable.
    Higher values indicate that the feature has a stronger influence on the model's predictions.
    This can help identify which factors are most crucial for hydrogen production in your system.
    """)
    importance = regr.feature_importances_
    feature_importance = pd.DataFrame(
        {'feature': feature_col, 'importance': importance})
    feature_importance = feature_importance.sort_values(
        'importance', ascending=False)

    fig_importance = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    fig_importance.update_layout(
        title='Feature Importance',
        xaxis_title='Importance',
        yaxis_title='Features',
        height=400
    )
    st.plotly_chart(fig_importance, use_container_width=True, key='gb_imp')

with tabs[2]:
    st.write(f"### Prophet Model Training and Prediction")
    with st.spinner('Training Prophet model... This may take a moment.'):
        prophet_data = df.rename(
            columns={'DateTime': 'ds', 'Hydrogen Production (kg)': 'y'})
        prophet_data = prophet_data.dropna(subset=['ds', 'y'] + feature_col)

        train_data, test_data = train_test_split(
            prophet_data, test_size=0.1, random_state=13)

        model = Prophet()
        for feature in feature_col:
            model.add_regressor(feature)
        model.add_seasonality(name='daily', period=1, fourier_order=4)
        model.fit(train_data)

        future = test_data[['ds'] + feature_col]
        forecast = model.predict(future)

        y_true = test_data['y']
        y_pred = forecast['yhat']

        df_cv = cross_validation(
            model, horizon='31 days', period='16 days', initial='300 days')
        df_p = performance_metrics(df_cv)
        df_p['horizon_days'] = df_p['horizon'].dt.total_seconds() / \
            (24 * 60 * 60)

    st.write("This chart visualizes various error metrics (RMSE, MAE, MAPE, MDAPE, and SMAPE) against the forecast horizon in days. It's crucial for assessing the model's predictive accuracy over time and understanding how the forecast quality degrades as we predict further into the future. ")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=df_p['horizon_days'], y=df_p['rmse'], mode='lines+markers', name='RMSE'))
    fig1.add_trace(go.Scatter(
        x=df_p['horizon_days'], y=df_p['mae'], mode='lines+markers', name='MAE'))
    fig1.add_trace(go.Scatter(
        x=df_p['horizon_days'], y=df_p['mape'], mode='lines+markers', name='MAPE'))
    fig1.add_trace(go.Scatter(
        x=df_p['horizon_days'], y=df_p['mdape'], mode='lines+markers', name='MDAPE'))
    fig1.add_trace(go.Scatter(
        x=df_p['horizon_days'], y=df_p['smape'], mode='lines+markers', name='SMAPE'))

    fig1.update_layout(
        title='RMSE, MAE, MAPE, MDAPE and SMAPE vs Forecast Horizon',
        xaxis_title='Horizon (days)',
        yaxis_title='Error',
        legend_title='Metric'
    )
    st.plotly_chart(fig1)

    colC, colD = st.columns(2, gap="large", vertical_alignment="center")

    with colC:
        st.subheader("Enter New Data for Prediction")
        solar_irradiance = st.number_input(
            "Solar Irradiance (W/m^2)", value=500)
        ambient_temperature = st.number_input(
            "Ambient Temperature (°C)", value=25)
        prediction_date = st.date_input("Prediction Date")

        new_data = pd.DataFrame({
            'ds': [prediction_date],
            'Solar Irradiance (W/m^2)': [solar_irradiance],
            'Ambient Temperature (°C)': [ambient_temperature]
        })

    with colD:
        prediction = model.predict(new_data)
        st.success(
            f"Predicted Hydrogen Production: {prediction['yhat'].values[0]:.2f} kg")

    st.subheader("Forecasted vs Actual")
    st.write("""
    The predicted vs actual chart compares the model's predictions with the actual observed values. 
    Points closer to the diagonal line indicate more accurate predictions. 
    Deviations from this line show where the model over- or under-predicts. 
    This visualization helps assess the model's overall performance and identify any systematic prediction errors.
    """)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=y_true,
        mode='markers',
        name='Actual',
        marker=dict(color='blue')
    ))

    # Add predicted values
    fig.add_trace(go.Scatter(
        x=test_data['ds'],
        y=y_pred,
        mode='lines',
        name='Predicted',
        line=dict(color='red')
    ))

    fig.update_layout(title='Hydrogen Production Forecast',
                      xaxis_title='Date', yaxis_title='Hydrogen Production (kg)')
    st.plotly_chart(fig)


''
''
st.divider()

st.header("Actual Hydrogen Production for 2023")

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
            tickfont=dict(color='#0f2113'),
            title_font=dict(color='#0f2113')
        )

        fig.update_yaxes(
            tickfont=dict(color='#0f2113'),
            title_font=dict(color='#0f2113')
        )

        st.plotly_chart(fig, use_container_width=True)
