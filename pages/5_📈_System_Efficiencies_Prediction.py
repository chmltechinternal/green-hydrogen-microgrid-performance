import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px
import math
import filters

st.set_page_config(
    page_title="Fuel Cell Efficiencies Prediction", layout="wide")

st.title("Fuel Cell Efficiencies Prediction")
st.write("""
This dashboard provides insights into the prediction of Fuel Cell Electrical Efficiency (%) using machine learning models. Fuel cell efficiency is a crucial factor in optimizing hydrogen consumption and improving energy performance.

Explore the visualizations to gain insights into performance metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²) to understand model accuracy and reliability.
""")


df = filters.load_data()
feature_col = ['Hydrogen Flow from Tank to FC (kg)']
output = ['Fuel Cell Electrical Efficiency (%)']

df['Hydrogen Consumption Rate (kg/kWh)'].unique()


def get_list_nnan(column_name):
    list_y_nan = list(
        df[column_name][df[column_name].apply(lambda x: math.isnan(x))].index)
    list_x_nan = []
    for i in feature_col:
        list_x_nan += list(df[i][df[i].apply(lambda x: math.isnan(x))].index)
    list_x_nan = list(set(list_x_nan))
    list_nan = list(set(list_x_nan+list_y_nan))
    list_notnan = [i for i in list(df.index) if i not in list_nan]
    return list_notnan


df['DateTime'] = pd.to_datetime(df['DateTime'])
data = df[['DateTime']+feature_col+output]
data.columns = ['ds', 'HyFlow', 'y']
train_data = data.iloc[:7000]
test_data = data.iloc[7000:]
X_train, X_test = train_data[['HyFlow']], test_data[['HyFlow']]
y_train, y_test = train_data['y'], test_data['y']

tab_titles = ["Gradient Boosting", "Random Forest", "Prophet"]
tabs = st.tabs(tab_titles)

params = {
    "n_estimators": 10,
    "max_depth": 10,
    "min_samples_split": 10,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)

with tabs[0]:
    st.write("### Gradient Boosting")
    st.write("In this tab, we use a Gradient Boosting Regressor to predict Fuel Cell Electrical Efficiency (%) based on the hydrogen flow rate. Gradient Boosting is an ensemble technique that builds multiple weak learners (decision trees) in sequence, improving predictions by correcting the errors made by previous models. The model’s performance is evaluated using key metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score. The plotted results display actual efficiency values alongside predicted values, giving insights into the model’s accuracy.")

    # mset = mean_squared_error(y_train, reg.predict(X_train))
    # print("The mean squared error (MSE) on train set: {:.4f}".format(mset))
    # mse = mean_squared_error(y_test, reg.predict(X_test))
    # print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

    # maet = mean_absolute_error(y_train, reg.predict(X_train))
    # print("The mean absolute error (MAE) on train set: {:.4f}".format(maet))
    # mae = mean_absolute_error(y_test, reg.predict(X_test))
    # print("The mean absolute error (MAE) on test set: {:.4f}".format(mae))

    # r2t = r2_score(y_train, reg.predict(X_train))
    # print("The R2 score on train set: {:.4f}".format(r2t))
    # r2 = r2_score(y_test, reg.predict(X_test))
    # print("The R2 score on test set: {:.4f}".format(r2))

    st.write("#### Model Metrics")
    metrics = {
        "MSE Train": mean_squared_error(y_train, reg.predict(X_train)),
        "MSE Test": mean_squared_error(y_test, reg.predict(X_test)),
        "MAE Train": mean_absolute_error(y_train, reg.predict(X_train)),
        "MAE Test": mean_absolute_error(y_test, reg.predict(X_test)),
        "R² Score Train": r2_score(y_train, reg.predict(X_train)),
        "R² Score Test": r2_score(y_test, reg.predict(X_test)),
    }
    num_metrics = len(metrics)

    # num_columns = (num_metrics + 1) // 2
    num_columns = 3
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

    train_data['yhat_train'] = reg.predict(X_train)
    test_data['yhat_test'] = reg.predict(X_test)
    forecaste_data = pd.concat(
        [train_data['yhat_train'], test_data['yhat_test']])
    forecaste_data_ds = pd.concat([train_data['ds'], test_data['ds']])

    fig = px.line()

    # Actual training data (blue)
    fig.add_scatter(x=train_data['ds'], y=y_train, mode='lines',
                    name='Actual (Train)', line=dict(color='blue'))

    # Actual test data (green)
    fig.add_scatter(x=test_data['ds'], y=y_test, mode='lines',
                    name='Actual (Test)', line=dict(color='green'))

    # Predicted data (red)
    fig.add_scatter(x=forecaste_data_ds, y=forecaste_data,
                    mode='lines', name='Predicted', line=dict(color='red'))

    fig.update_layout(
        title='Actual vs Predicted Fuel Cell Electrical Efficiency',
        xaxis_title='Date',
        yaxis_title='Fuel Cell Electrical Efficiency (%)',
        legend_title='Legend',
        template='plotly_white'
    )
    ''
    ''
    st.plotly_chart(fig)

with tabs[1]:
    st.write("### Random Forest")
    st.write("This tab presents predictions made using a Random Forest Regressor, another ensemble learning method that constructs multiple decision trees to improve prediction stability and accuracy. Random Forest is effective at handling non-linear data and reducing overfitting. The model’s performance is evaluated with MSE, MAE, and R² score. The plotted graph compares the predicted efficiency against actual data, showing how well the model aligns with observed values.")
    regr = RandomForestRegressor(
        n_estimators=10, max_depth=10, random_state=13)
    regr.fit(X_train, y_train)

    st.write("#### Model Metrics")
    metrics = {
        "MSE Train": mean_squared_error(y_train, regr.predict(X_train)),
        "MSE Test": mean_squared_error(y_test, regr.predict(X_test)),
        "MAE Train": mean_absolute_error(y_train, regr.predict(X_train)),
        "MAE Test": mean_absolute_error(y_test, regr.predict(X_test)),
        "R² Score Train": r2_score(y_train, regr.predict(X_train)),
        "R² Score Test": r2_score(y_test, regr.predict(X_test)),
    }
    num_metrics = len(metrics)

    # num_columns = (num_metrics + 1) // 2
    num_columns = 3
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

    train_data['yhat_train'] = regr.predict(X_train)
    test_data['yhat_test'] = regr.predict(X_test)

    forecaste_data = pd.concat(
        [train_data['yhat_train'], test_data['yhat_test']])
    forecaste_data_ds = pd.concat([train_data['ds'], test_data['ds']])

    fig = px.line()

    # Actual training data (blue)
    fig.add_scatter(x=train_data['ds'], y=y_train, mode='lines',
                    name='Actual (Train)', line=dict(color='blue'))

    # Actual test data (green)
    fig.add_scatter(x=test_data['ds'], y=y_test, mode='lines',
                    name='Actual (Test)', line=dict(color='green'))

    # Predicted data (red)
    fig.add_scatter(x=forecaste_data_ds, y=forecaste_data,
                    mode='lines', name='Predicted', line=dict(color='red'))

    fig.update_layout(
        title='Actual vs Predicted Fuel Cell Electrical Efficiency with Random Forest',
        xaxis_title='Date',
        yaxis_title='Fuel Cell Electrical Efficiency (%)',
        legend_title='Legend',
        template='plotly_white'
    )
    ''
    ''
    st.plotly_chart(fig)

with tabs[2]:
    st.write("### Prophet")
    st.write("In this tab, we use Facebook Prophet, a robust time series forecasting tool designed for data with strong seasonal patterns. Prophet models the efficiency trend over time, incorporating external factors like the hydrogen flow rate. It generates future forecasts along with an uncertainty interval, providing insights into possible variations. The graph displays actual data and predicted efficiency trends, showcasing the model’s ability to capture seasonal and long-term patterns effectively.")

    model = Prophet()
    # model.fit(train_data[['ds','y']])
    # model = Prophet(growth='logistic')
    # Add the external regressor(s) to the model
    model.add_regressor('HyFlow')
    model.add_seasonality(name='daily', period=1, fourier_order=5)
    model.fit(train_data)
    test_p = data[['ds', 'HyFlow']]

    forecast = model.predict(test_p)

    # Calculate  mean squared error
    print('Train MSE: %f' %
          np.mean((forecast.loc[:7000, 'yhat']-train_data['y'])**2))
    # Calculate  mean absolute error
    print('Train MAE: %f' % mean_absolute_error(
        train_data['y'], forecast.loc[:6999, 'yhat']))
    # Calculate R2 score
    print('Train R2: %f' % r2_score(
        train_data['y'], forecast.loc[:6999, 'yhat']))
    # Calculate  mean squared error
    print('Test MSE: %f' %
          np.mean((forecast.loc[7000:, 'yhat']-test_data['y'])**2))
    # Calculate  mean absolute error
    print('Train MAE: %f' % mean_absolute_error(
        test_data['y'], forecast.loc[7000:, 'yhat']))
    # Calculate R2 score
    print('Train R2: %f' % r2_score(
        test_data['y'], forecast.loc[7000:, 'yhat']))

    st.write("#### Model Metrics")
    metrics = {
        "MSE Train": np.mean((forecast.loc[:7000, 'yhat']-train_data['y'])**2),
        "MSE Test": np.mean((forecast.loc[7000:, 'yhat']-test_data['y'])**2),
        "MAE Train": mean_absolute_error(train_data['y'], forecast.loc[:6999, 'yhat']),
        "MAE Test": mean_absolute_error(test_data['y'], forecast.loc[7000:, 'yhat']),
        "R² Score Train": r2_score(train_data['y'], forecast.loc[:6999, 'yhat']),
        "R² Score Test": r2_score(test_data['y'], forecast.loc[7000:, 'yhat']),
    }
    num_metrics = len(metrics)

    # num_columns = (num_metrics + 1) // 2
    num_columns = 3
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

    fig = px.line()

    # Actual training data (blue)
    fig.add_scatter(x=train_data['ds'], y=y_train, mode='lines',
                    name='Actual (Train)', line=dict(color='blue'))

    # Actual test data (green)
    fig.add_scatter(x=test_data['ds'], y=y_test, mode='lines',
                    name='Actual (Test)', line=dict(color='green'))

    # Predicted data (red)
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat'],
                    mode='lines', name='Predicted', line=dict(color='red'))

    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist(
        ) + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255,192,203,0.3)',
        line=dict(color='rgba(255,192,203,0.3)'),
        hoverinfo="skip",
        showlegend=True,
        name='Uncertainty Interval',
    ))

    fig.update_layout(
        title='Actual vs Predicted Fuel Cell Electrical Efficiency with Prophet',
        xaxis_title='Date',
        yaxis_title='Fuel Cell Electrical Efficiency (%)',
        legend_title='Legend',
        template='plotly_white'
    )
    ''
    ''
    st.plotly_chart(fig)
