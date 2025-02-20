import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import filters

st.set_page_config(page_title="PV Production Prediction", layout="wide")

st.title("PV Production Prediction")
st.write("""
The Random Forest model is used here to predict PV Production for a microgrid system using the features - 'Solar Irradiance (W/m^2)', 'Electrolyser Power (kW)', 'Ambient Temperature (°C)', 'Electrolyzer Losses (kWh)', and 'Hour'. 
The following metrics measure the performance of the model - Mean Absolute Error, Mean Square Error, Mean Absolute Percentage Error and R² Score. 
Modelling was done on the full dataset provided as well as daily averages of the data.
""")

df = filters.load_data()
df['DayOfYear'] = df['DateTime'].dt.dayofyear
# st.divider()
tab_titles = ['Full Dataset', 'Daily Averages']
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.header("Prediction with Full Dataset")
    st.subheader("Dataset Overview")
    st.write("Here's a glimpse of the data we're working with:")
    st.dataframe(df[['DateTime', 'Month', 'Hour', 'Solar Irradiance (W/m^2)',
                    'Electrolyser Power (kW)', 'PV Production (kWh)',
                     'Ambient Temperature (°C)',
                     'Electrolyzer Losses (kWh)']].head())

    def calculate_daily_averages(df):
        df = df.set_index('DateTime')
        daily_averages = df.resample('D').mean()
        daily_averages = daily_averages.reset_index()

        return daily_averages

    daily_avg = calculate_daily_averages(
        df[['DateTime', 'Solar Irradiance (W/m^2)',
            'Electrolyser Power (kW)', 'PV Production (kWh)',
            'Ambient Temperature (°C)',
            'Electrolyzer Losses (kWh)']])
    daily_avg['DayOfYear'] = daily_avg['DateTime'].dt.dayofyear
    daily_avg['Month'] = daily_avg['DateTime'].dt.month

    features = [
        'Solar Irradiance (W/m^2)',
        'Electrolyser Power (kW)',
        'Ambient Temperature (°C)',
        'Electrolyzer Losses (kWh)',
        'Hour'
    ]
    output = ['PV Production (kWh)']

    # Calculate daily averages

    X_full = df[features]
    y_full = df['PV Production (kWh)']

    X_full_train, X_full_test, y_full_train, y_full_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)
    X_full_test_scaled = scaler.transform(X_full_test)

    # Parameter tuning for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    ''
    ''
    st.write("#### Parameter Selection")
    rf = RandomForestRegressor(random_state=42)
    # grid_search = GridSearchCV(
    #     estimator=rf,
    #     param_grid=param_grid,
    #     cv=5,
    #     n_jobs=-1,
    #     scoring='r2',
    #     verbose=1
    # )

    # grid_search.fit(X_train_scaled, y_train)
    full_best_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 4
    }
    st.write('After parameter tuning with GridSearchCV, the following were defined as the best parameters for the Random Forest model:')
    for parameter, value in full_best_params.items():
        st.markdown(f"**{parameter}:** {value}")

    rf_final = RandomForestRegressor(
        **full_best_params,
        random_state=42
    )
    rf_final.fit(X_full_train_scaled, y_full_train)
    # Predictions
    y_full_pred_rf = rf_final.predict(X_full_test_scaled)

    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    errors_full_rf = y_full_test - y_full_pred_rf
    ''
    ''
    st.write("#### Model Metrics")
    metrics = {
        "MSE": mean_squared_error(y_full_test, y_full_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_full_test, y_full_pred_rf)),
        "MAE": mean_absolute_error(y_full_test, y_full_pred_rf),
        "MAPE": mean_absolute_percentage_error(y_full_test, y_full_pred_rf),
        "R² Score": r2_score(y_full_test, y_full_pred_rf),
    }
    num_metrics = len(metrics)

    # num_columns = (num_metrics + 1) // 2
    num_columns = 5
    cols = st.columns(num_columns, gap='large')

    for i, (metric_name, value) in enumerate(metrics.items()):
        # best_value = best_metrics[metric_name]['value']
        color = "#90EE90"
        # color = "#90EE90" if np.isclose(value, best_value) else "#f0f0f0"
        col_index = i % num_columns
        # col_index = i // 2

        with cols[col_index]:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0; color: #000000; text-align: center">
                    <strong>{metric_name}</strong> 
                    <p style="padding: 0; margin: 0">{value:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    def plot_feature_importance(features, importances):
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=True)

        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                     title='Random Forest Feature Importance',
                     labels={'importance': 'Importance Score',
                             'feature': 'Features'},
                     color='importance', color_continuous_scale='Greens')

        fig.update_layout(height=600, width=800)
        st.plotly_chart(fig)

    def plot_actual_vs_predicted(y_test, y_pred_rf):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=y_test, y=y_pred_rf, mode='markers',
                                 marker=dict(color='green', opacity=0.5),
                                 name='Predictions'))

        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                 y=[y_test.min(), y_test.max()],
                                 mode='lines', line=dict(color='red', dash='dash'),
                                 name='Perfect Prediction'))

        fig.update_layout(title='Random Forest: Actual vs Predicted PV Production',
                          xaxis_title='Actual PV Production',
                          yaxis_title='Predicted PV Production',
                          height=600, width=800)

        st.plotly_chart(fig)

    def plot_error_distribution(errors_rf, y_pred_rf):
        fig = go.Figure()

        # Error histogram
        fig.add_trace(go.Histogram(x=errors_rf, nbinsx=50, name='Error Distribution',
                                   marker_color='lightgreen', marker_line_color='black',
                                   marker_line_width=1))

        fig.add_vline(x=0, line_dash="dash", line_color="red")

        fig.update_layout(title='Distribution of RF Prediction Errors',
                          xaxis_title='Error (kWh)',
                          yaxis_title='Frequency',
                          height=400, width=600)

        st.plotly_chart(fig)

        fig = px.scatter(x=y_pred_rf, y=errors_rf, opacity=0.5,
                         labels={
                             'x': 'Predicted PV Production (kWh)', 'y': 'Error (kWh)'},
                         title='RF Prediction Errors vs Predicted Values')

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        fig.update_layout(height=400, width=600)

        st.plotly_chart(fig)

    importances = rf_final.feature_importances_
    ''
    ''
    st.subheader("Feature Importance")
    plot_feature_importance(features, importances)

    st.subheader("Actual vs Predicted")
    plot_actual_vs_predicted(y_full_test, y_full_pred_rf)

    st.subheader("Error Analysis")
    plot_error_distribution(errors_full_rf, y_full_pred_rf)

with tabs[1]:
    st.header("Prediction with Daily Averages")
    st.subheader("Dataset Overview")
    st.write("Here's a glimpse of the daily averages of the data we're working with:")

    st.dataframe(daily_avg.head())

    daily_features = [
        'Solar Irradiance (W/m^2)',
        'Electrolyser Power (kW)',
        'Ambient Temperature (°C)',
        'Electrolyzer Losses (kWh)',
        'DayOfYear',
        'Month'
    ]

    X_daily = daily_avg[daily_features]
    y_daily = daily_avg['PV Production (kWh)']

    # Split data
    X_daily_train, X_daily_test, y_daily_train, y_daily_test = train_test_split(
        X_daily, y_daily, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_daily_train_scaled = scaler.fit_transform(X_daily_train)
    X_daily_test_scaled = scaler.transform(X_daily_test)

    # # Parameter grid for random search
    # param_distributions = {
    #     'n_estimators': [100, 200, 300],
    #     'max_depth': [10, 15, 20, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }

    # # Random search CV
    # random_search = RandomizedSearchCV(
    #     estimator=RandomForestRegressor(random_state=42),
    #     param_distributions=param_distributions,
    #     n_iter=20,
    #     cv=5,
    #     n_jobs=-1,
    #     scoring='r2',
    #     verbose=1,
    #     random_state=42
    # )

    # random_search.fit(X_train_scaled, y_train)
    # print(f"Best parameters: {random_search.best_params_}")

    daily_best_params = {'n_estimators': 200,
                         'min_samples_split': 10, 'min_samples_leaf': 4, 'max_depth': 20}

    st.write('After parameter tuning with RandomSearchCV, the following were defined as the best parameters for the Random Forest model:')
    for parameter, value in daily_best_params.items():
        st.markdown(f"**{parameter}:** {value}")

    # Final model with best parameters
    rf_daily = RandomForestRegressor(
        **daily_best_params,
        random_state=42
    )

    rf_daily.fit(X_daily_train_scaled, y_daily_train)

    y_daily_pred_rf = rf_daily.predict(X_daily_test_scaled)

    ''
    ''
    st.write("#### Model Metrics")
    daily_metrics = {
        "MSE": mean_squared_error(y_daily_test, y_daily_pred_rf),
        "RMSE": np.sqrt(mean_squared_error(y_daily_test, y_daily_pred_rf)),
        "MAE": mean_absolute_error(y_daily_test, y_daily_pred_rf),
        "MAPE": mean_absolute_percentage_error(y_daily_test, y_daily_pred_rf),
        "R² Score": r2_score(y_daily_test, y_daily_pred_rf),
    }
    num_metrics = len(metrics)

    # num_columns = (num_metrics + 1) // 2
    num_columns = 5
    cols = st.columns(num_columns, gap='large')

    for i, (metric_name, value) in enumerate(daily_metrics.items()):
        # best_value = best_metrics[metric_name]['value']
        color = "#90EE90"
        # color = "#90EE90" if np.isclose(value, best_value) else "#f0f0f0"
        col_index = i % num_columns
        # col_index = i // 2

        with cols[col_index]:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0; color: #000000; text-align: center">
                    <strong>{metric_name}</strong> 
                    <p style="padding: 0; margin: 0">{value:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    daily_importances = rf_daily.feature_importances_
    ''
    ''
    st.subheader("Feature Importance (Daily Average)")
    plot_feature_importance(daily_features, daily_importances)

    st.subheader("Actual vs Predicted (Daily Average)")
    plot_actual_vs_predicted(y_daily_test, y_daily_pred_rf)
