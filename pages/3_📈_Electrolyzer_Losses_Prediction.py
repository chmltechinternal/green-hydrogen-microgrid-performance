import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet import Prophet
import plotly.graph_objects as go
import math
import filters

st.set_page_config(page_title="Electrolyzer Losses Prediction", layout="wide")

st.title("Electrolyzer Losses Prediction")
st.write("""
This application compares the performance of different machine learning models for predicting electrolyzer losses for a microgrid system based on Hydrogen production and Energy supplied to Electrolyzer (E_PV2EZ).
We are only concerned with the metrics - Mean Absolute Error, Mean Square Error and R² Score. 

After training, the model metrics that show the best performance are highlighted in green (minimum error values and maximum R² score values).
""")

# Load and display data
df = filters.load_data()

# Dataset overview
st.header("Dataset Overview")
st.write("Here's a glimpse of the data we're working with:")
st.dataframe(df[['DateTime', 'Month', 'Hour', 'Electrolyzer Losses (kWh)',
             'E_PV2EZ (kWh)', 'Hydrogen Production (kg)']].head())


def calculate_daily_averages(df):
    df = df.set_index('DateTime')

    # Calculate daily averages
    # daily_averages = df.resample('D').agg(['mean', 'median', 'std'])
    daily_averages = df.resample('D').mean()

    # Reset index to make DateTime a column again
    daily_averages = daily_averages.reset_index()

    return daily_averages


feature_col = ['E_PV2EZ (kWh)', 'Hydrogen Production (kg)']
output = ['Electrolyzer Losses (kWh)']

# Calculate daily averages
daily_avg = calculate_daily_averages(
    df[['DateTime', 'Electrolyzer Losses (kWh)', 'E_PV2EZ (kWh)', 'Hydrogen Production (kg)']])

# Display the daily averages
st.dataframe(daily_avg.head())

st.divider()


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


list_notnan = get_list_nnan('Electrolyzer Losses (kWh)')
X = df[feature_col].loc[list_notnan]
y = df['Electrolyzer Losses (kWh)'].loc[list_notnan]

# X_daily = daily_avg[feature_col].loc[list_notnan]
# y_daily = daily_avg['Electrolyzer Losses (kWh)'].loc[list_notnan]


data = df[['DateTime']+feature_col+output]
data.columns = ['ds', 'E_PV2EZ', 'Hydrogen', 'y']
train_data = data.iloc[:7000]
test_data = data.iloc[7000:]
X_train, X_test = train_data[['E_PV2EZ', 'Hydrogen']
                             ], test_data[['E_PV2EZ', 'Hydrogen']]
y_train, y_test = train_data['y'], test_data['y']

model_metrics = {}


def evaluate_best_metrics(model_metrics):
    best_metrics = {
        'R2_train': {'value': -np.inf, 'model': ''},
        'R2_test': {'value': -np.inf, 'model': ''},
        'MSE_train': {'value': np.inf, 'model': ''},
        'MSE_test': {'value': np.inf, 'model': ''},
        'MAE_train': {'value': np.inf, 'model': ''},
        'MAE_test': {'value': np.inf, 'model': ''}
    }

    for model, metrics in model_metrics.items():
        for metric, value in metrics.items():
            if metric.startswith('R2'):
                if value > best_metrics[metric]['value']:
                    best_metrics[metric]['value'] = value
                    best_metrics[metric]['model'] = model
            else:
                if value < best_metrics[metric]['value']:
                    best_metrics[metric]['value'] = value
                    best_metrics[metric]['model'] = model

    return best_metrics


def display_metrics(model_name, model_metrics, best_metrics):
    metrics = model_metrics[model_name]
    num_metrics = len(metrics)

    cols = st.columns(num_metrics, gap='large')

    for i, (metric_name, value) in enumerate(metrics.items()):
        best_value = best_metrics[metric_name]['value']
        color = "#90EE90" if np.isclose(value, best_value) else "#f0f0f0"
        col_index = i % num_metrics

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


def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """Trains a model, makes predictions, and calculates performance metrics."""
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "MSE_train": mean_squared_error(y_train, y_train_pred),
        "MSE_test": mean_squared_error(y_test, y_test_pred),
        "MAE_train": mean_absolute_error(y_train, y_train_pred),
        "MAE_test": mean_absolute_error(y_test, y_test_pred),
        "R2_train": r2_score(y_train, y_train_pred),
        "R2_test": r2_score(y_test, y_test_pred),
    }

    return metrics, y_train_pred, y_test_pred


def plot_predict_v_actual(model_name, train_data, test_data, y_train, y_train_pred, y_test_pred):
    train_data["yhat_train"] = y_train_pred
    test_data["yhat_test"] = y_test_pred
    forecaste_data = pd.concat(
        [train_data["yhat_train"], test_data["yhat_test"]])
    forecaste_data_ds = pd.concat([train_data["ds"], test_data["ds"]])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=train_data["ds"],
            y=y_train,
            mode="lines",
            name="Actual Electrolyzer Losses (Train)",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_data["ds"],
            y=y_test,
            mode="lines",
            name="Actual Electrolyzer Losses (Test)",
            line=dict(color="green"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecaste_data_ds,
            y=forecaste_data,
            mode="lines",
            name="Predicted Electrolyzer Losses",
            line=dict(color="red"),
        )
    )

    fig.update_layout(
        title=f"Electrolyzer Losses Prediction with {model_name}",
        xaxis_title="Date",
        yaxis_title="Electrolyzer Losses",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(
            title=dict(
                text="Date",
                standoff=45
            )
        ),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    st.plotly_chart(fig, use_container_width=True)


################### RANDOM FOREST ##########################
regr = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=13)
model_metrics['Random Forest'], rf_y_train_pred, rf_y_test_pred = train_and_evaluate(
    regr, X_train, y_train, X_test, y_test)

############### GRADIENT BOOST #################################
params = {
    "n_estimators": 10,
    "max_depth": 10,
    "min_samples_split": 10,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = GradientBoostingRegressor(**params)
model_metrics['Gradient Boosting'], gb_y_train_pred, gb_y_test_pred = train_and_evaluate(
    reg, X_train, y_train, X_test, y_test)

############ PROPHET ######################################
model = Prophet()
# model.fit(train_data[['ds','y']])
# model = Prophet(growth='logistic')
# Add the external regressor(s) to the model
model.add_regressor('E_PV2EZ')
model.add_regressor('Hydrogen')
model.add_seasonality(name='daily', period=1, fourier_order=5)
model.fit(train_data)
# test_pred = model.make_future_dataframe(periods=1760, freq='h')
test_p = data[['ds', 'E_PV2EZ', 'Hydrogen']]

forecast = model.predict(test_p)

# Calculate metrics for Prophet
model_metrics['Prophet'] = {
    "MSE_train": np.mean((forecast.loc[:6999, "yhat"] - train_data["y"]) ** 2),
    "MSE_test": np.mean((forecast.loc[7000:, "yhat"] - test_data["y"]) ** 2),
    "MAE_train": mean_absolute_error(train_data["y"], forecast.loc[:6999, "yhat"]),
    "MAE_test": mean_absolute_error(test_data["y"], forecast.loc[7000:, "yhat"]),
    "R2_train": r2_score(train_data["y"], forecast.loc[:6999, "yhat"]),
    "R2_test": r2_score(test_data["y"], forecast.loc[7000:, "yhat"]),
}

best_metrics = evaluate_best_metrics(model_metrics)

st.write('#### Random Forest Model Training')
display_metrics('Random Forest', model_metrics, best_metrics)
plot_predict_v_actual('Random Forest', train_data, test_data,
                      y_train, rf_y_train_pred, rf_y_test_pred)

st.divider()

st.write('#### Gradient Boosting Model Training')
display_metrics('Gradient Boosting', model_metrics, best_metrics)
plot_predict_v_actual('Gradient Boosting', train_data,
                      test_data, y_train, gb_y_train_pred, gb_y_test_pred)

st.divider()

st.write('#### Prophet Model Training')
display_metrics('Prophet', model_metrics, best_metrics)

prophet_fig = go.Figure()

prophet_fig.add_trace(go.Scatter(x=train_data['ds'], y=y_train,
                                 mode='lines', name='Actual Electrolyzer Losses (Train)',
                                 line=dict(color='blue')))

prophet_fig.add_trace(go.Scatter(x=test_data['ds'], y=y_test,
                                 mode='lines', name='Actual Electrolyzer Losses (Test)',
                                 line=dict(color='green')))

prophet_fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                 mode='lines', name='Predicted Electrolyzer Losses',
                                 line=dict(color='red')))

prophet_fig.add_trace(go.Scatter(
    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255,192,203,0.3)',
    line=dict(color='rgba(255,192,203,0.3)'),
    hoverinfo="skip",
    showlegend=True,
    name='Uncertainty Interval',
))

prophet_fig.update_layout(
    title='Electrolyzer Losses Prediction with Prophet',
    xaxis_title='Date',
    yaxis_title='Electrolyzer Losses',
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.2,
        xanchor="center",
        x=0.5
    ),
    xaxis=dict(
        title=dict(
            text="Date",
            standoff=45
        )
    ),
)

prophet_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
prophet_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')

st.plotly_chart(prophet_fig, use_container_width=True)
st.divider()
