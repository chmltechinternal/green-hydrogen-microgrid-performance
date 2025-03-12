import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import filters

st.set_page_config(page_title="PV Production Prediction", layout="wide")

st.title("PV Production Prediction")
st.write("""
The Random Forest model is used here to predict PV Production for a microgrid system. 
The following metrics measure the performance of the model - Mean Absolute Error, Mean Square Error, Mean Absolute Percentage Error and R² Score. 
Modelling was done on the full dataset provided and incorporates solar condition categories to improve prediction accuracy.
""")

df = filters.load_data()
df['DayOfYear'] = df['DateTime'].dt.dayofyear

# Create solar condition categories based on Solar Irradiance
solar_quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
solar_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']

# Calculate quantile thresholds based on positive irradiance values
day_irradiance = df[df['Solar Irradiance (W/m^2)'] > 10]['Solar Irradiance (W/m^2)']
quantile_thresholds = day_irradiance.quantile(solar_quantiles).values

# Create a function to categorize solar conditions
def categorize_solar_condition(irradiance):
    if irradiance <= 10:  # Night time or very cloudy
        return 'Night/Overcast'
    elif irradiance <= quantile_thresholds[1]:
        return 'Very Low'
    elif irradiance <= quantile_thresholds[2]:
        return 'Low'
    elif irradiance <= quantile_thresholds[3]:
        return 'Medium'
    elif irradiance <= quantile_thresholds[4]:
        return 'High'
    else:
        return 'Very High'

# Apply categorization to create a new column
df['Solar_Condition'] = df['Solar Irradiance (W/m^2)'].apply(categorize_solar_condition)

# Display thresholds for solar conditions
with st.expander("Solar Condition Thresholds"):
    st.write("Solar conditions are categorized based on the following irradiance thresholds:")
    for i, label in enumerate(solar_labels):
        if i < len(solar_labels) - 1:
            st.write(f"**{label}**: {quantile_thresholds[i]:.2f} to {quantile_thresholds[i+1]:.2f} W/m²")
        else:
            st.write(f"**{label}**: > {quantile_thresholds[i]:.2f} W/m²")
    st.write(f"**Night/Overcast**: ≤ 10 W/m²")

# Convert categorical variable to one-hot encoding
df_encoded = pd.get_dummies(df, columns=['Solar_Condition'], prefix='SolarCond')

# Add a new tab for the enhanced model with solar conditions
tab_titles = ['Original Model', 'Enhanced Model with Solar Conditions', 'Interactive Prediction']
tabs = st.tabs(tab_titles)

with tabs[0]:
    st.header("Original PV Production Model")
    st.subheader("Dataset Overview")
    st.write("Here's a glimpse of the data we're working with:")
    st.dataframe(df[['DateTime', 'Month', 'Hour', 'Solar Irradiance (W/m^2)',
                    'Electrolyser Power (kW)', 'PV Production (kWh)',
                     'Ambient Temperature (°C)',
                     'Electrolyzer Losses (kWh)']].head())

    features = [
        'Solar Irradiance (W/m^2)',
        'Electrolyser Power (kW)',
        'Ambient Temperature (°C)',
        'Electrolyzer Losses (kWh)',
        'Hour'
    ]
    
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
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

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

    num_columns = 5
    cols = st.columns(num_columns, gap='large')

    for i, (metric_name, value) in enumerate(metrics.items()):
        color = "#90EE90"
        col_index = i % num_columns

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
    st.header("Enhanced Model with Solar Conditions")
    st.subheader("Dataset Overview with Solar Conditions")
    
    # Display the original data with solar conditions
    st.write("Here's the data with added Solar Condition categories:")
    st.dataframe(df[['DateTime', 'Month', 'Hour', 'Solar Irradiance (W/m^2)',
                    'Solar_Condition', 'PV Production (kWh)',
                    'Ambient Temperature (°C)',
                    'Electrolyzer Losses (kWh)']].head(10))
    
    # Display solar condition distribution
    st.write("#### Distribution of Solar Conditions")
    solar_condition_counts = df['Solar_Condition'].value_counts().reset_index()
    solar_condition_counts.columns = ['Solar Condition', 'Count']
    
    # Create a color map for the conditions
    condition_categories = ['Night/Overcast', 'Very Low', 'Low', 'Medium', 'High', 'Very High']
    colors = ['darkblue', 'blue', 'skyblue', 'greenyellow', 'orange', 'red']
    condition_color_map = dict(zip(condition_categories, colors))
    
    # Plot distribution
    fig = px.bar(solar_condition_counts, x='Solar Condition', y='Count',
                 color='Solar Condition',
                 color_discrete_map=condition_color_map,
                 title='Distribution of Solar Conditions')
    st.plotly_chart(fig)
    
    # Define enhanced features with solar conditions
    enhanced_features = [
        'Solar Irradiance (W/m^2)', 
        'Electrolyser Power (kW)', 
        'Ambient Temperature (°C)', 
        'Electrolyzer Losses (kWh)',
        'Hour',
        'SolarCond_Very Low',
        'SolarCond_Low',
        'SolarCond_Medium',
        'SolarCond_High',
        'SolarCond_Very High',
        'SolarCond_Night/Overcast'
    ]
    
    X_enhanced = df_encoded[enhanced_features]
    y_enhanced = df_encoded['PV Production (kWh)']
    
    X_enhanced_train, X_enhanced_test, y_enhanced_train, y_enhanced_test = train_test_split(
        X_enhanced, y_enhanced, test_size=0.2, random_state=42)
    
    # Scale numeric features only
    numeric_features = [
        'Solar Irradiance (W/m^2)', 
        'Electrolyser Power (kW)', 
        'Ambient Temperature (°C)', 
        'Electrolyzer Losses (kWh)',
        'Hour'
    ]
    categorical_features = [col for col in enhanced_features if col.startswith('SolarCond_')]
    
    # Create copies for scaling
    X_enhanced_train_scaled = X_enhanced_train.copy()
    X_enhanced_test_scaled = X_enhanced_test.copy()
    
    # Apply scaling only to numeric features
    scaler_enhanced = StandardScaler()
    X_enhanced_train_scaled[numeric_features] = scaler_enhanced.fit_transform(X_enhanced_train[numeric_features])
    X_enhanced_test_scaled[numeric_features] = scaler_enhanced.transform(X_enhanced_test[numeric_features])
    
    st.write("#### Parameter Selection")
    enhanced_best_params = {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 4
    }
    
    st.write('The enhanced model uses the following parameters:')
    for parameter, value in enhanced_best_params.items():
        st.markdown(f"**{parameter}:** {value}")
    
    # Train enhanced model
    rf_enhanced = RandomForestRegressor(
        **enhanced_best_params,
        random_state=42
    )
    
    rf_enhanced.fit(X_enhanced_train_scaled, y_enhanced_train)
    y_enhanced_pred = rf_enhanced.predict(X_enhanced_test_scaled)
    
    errors_enhanced = y_enhanced_test - y_enhanced_pred
    
    # Calculate metrics for enhanced model
    st.write("#### Model Metrics")
    enhanced_metrics = {
        "MSE": mean_squared_error(y_enhanced_test, y_enhanced_pred),
        "RMSE": np.sqrt(mean_squared_error(y_enhanced_test, y_enhanced_pred)),
        "MAE": mean_absolute_error(y_enhanced_test, y_enhanced_pred),
        "MAPE": mean_absolute_percentage_error(y_enhanced_test, y_enhanced_pred),
        "R² Score": r2_score(y_enhanced_test, y_enhanced_pred),
    }
    
    cols = st.columns(5, gap='large')
    for i, (metric_name, value) in enumerate(enhanced_metrics.items()):
        color = "#90EE90"
        with cols[i % 5]:
            st.markdown(
                f"""
                <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0; color: #000000; text-align: center">
                    <strong>{metric_name}</strong> 
                    <p style="padding: 0; margin: 0">{value:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Feature importance for enhanced model
    enhanced_importances = rf_enhanced.feature_importances_
    st.subheader("Feature Importance")
    plot_feature_importance(enhanced_features, enhanced_importances)
    
    # Actual vs Predicted
    st.subheader("Actual vs Predicted")
    
    # Add solar condition info for visualization
    X_enhanced_test_with_pred = X_enhanced_test.copy()
    X_enhanced_test_with_pred['Actual'] = y_enhanced_test
    X_enhanced_test_with_pred['Predicted'] = y_enhanced_pred
    
    # Determine the dominant solar condition for each data point
    solar_condition_cols = [col for col in X_enhanced_test_with_pred.columns if col.startswith('SolarCond_')]
    X_enhanced_test_with_pred['Dominant_Condition'] = X_enhanced_test_with_pred[solar_condition_cols].idxmax(axis=1)
    X_enhanced_test_with_pred['Dominant_Condition'] = X_enhanced_test_with_pred['Dominant_Condition'].str.replace('SolarCond_', '')
    
    # Plot colored by solar condition
    fig = go.Figure()
    
    for condition in condition_categories:
        subset = X_enhanced_test_with_pred[X_enhanced_test_with_pred['Dominant_Condition'] == condition]
        if not subset.empty:
            fig.add_trace(go.Scatter(
                x=subset['Actual'],
                y=subset['Predicted'],
                mode='markers',
                marker=dict(color=condition_color_map[condition], opacity=0.6),
                name=condition
            ))
    
    fig.add_trace(go.Scatter(
        x=[y_enhanced_test.min(), y_enhanced_test.max()],
        y=[y_enhanced_test.min(), y_enhanced_test.max()],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title='Enhanced Model: Actual vs Predicted PV Production by Solar Condition',
        xaxis_title='Actual PV Production (kWh)',
        yaxis_title='Predicted PV Production (kWh)',
        height=600,
        width=800
    )
    
    st.plotly_chart(fig)
    
    # Error analysis by solar condition
    error_by_condition = X_enhanced_test_with_pred.groupby('Dominant_Condition').agg({
        'Actual': ['mean', 'count'],
        'Predicted': 'mean',
        'Actual': lambda x: mean_absolute_error(x, X_enhanced_test_with_pred.loc[x.index, 'Predicted'])
    })
    
    error_by_condition.columns = ['MAE', 'Count', 'Avg Predicted']
    
    st.subheader("Error Analysis by Solar Condition")
    st.dataframe(error_by_condition)
    
    # Plot MAE by solar condition
    fig = px.bar(
        error_by_condition.reset_index(),
        x='Dominant_Condition',
        y='MAE',
        color='Dominant_Condition',
        color_discrete_map=condition_color_map,
        title='Mean Absolute Error by Solar Condition',
        labels={'Dominant_Condition': 'Solar Condition', 'MAE': 'Mean Absolute Error (kWh)'}
    )
    
    # Add count labels
    for i, condition in enumerate(error_by_condition.index):
        fig.add_annotation(
            x=condition,
            y=error_by_condition.loc[condition, 'MAE'] + 5,
            text=f"n={error_by_condition.loc[condition, 'Count']}",
            showarrow=False
        )
    
    st.plotly_chart(fig)
    
    # Error distribution
    st.subheader("Error Distribution")
    plot_error_distribution(errors_enhanced, y_enhanced_pred)

with tabs[2]:
    st.header("Interactive PV Production Prediction")
    st.write("""
    Adjust the input parameters below to predict PV production for specific conditions.
    The model will use the enhanced Random Forest model with solar condition categories.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Select Parameters")
        
        # Create month filter
        all_months = sorted(df['Month'].unique())
        selected_month = st.selectbox("Month", options=all_months)
        
        # Create solar condition filter
        selected_condition = st.selectbox(
            "Solar Condition", 
            options=condition_categories,
            format_func=lambda x: x
        )
        
        # Get typical irradiance for the selected condition
        if selected_condition == "Night/Overcast":
            typical_irradiance = 5.0  # Default for night/overcast
        else:
            condition_index = solar_labels.index(selected_condition)
            if condition_index < len(solar_labels) - 1:
                typical_irradiance = (quantile_thresholds[condition_index] + quantile_thresholds[condition_index + 1]) / 2
            else:
                typical_irradiance = quantile_thresholds[condition_index] * 1.1
        
        # Input features sliders
        irradiance = st.slider(
            "Solar Irradiance (W/m²)", 
            min_value=0.0,
            max_value=1200.0,
            value=float(typical_irradiance),
            step=10.0
        )
        
        electrolyser_power = st.slider(
            "Electrolyser Power (kW)",
            min_value=0.0,
            max_value=1500.0,
            value=200.0,
            step=50.0
        )
        
        ambient_temp = st.slider(
            "Ambient Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=25.0,
            step=1.0
        )
        
        electrolyzer_losses = st.slider(
            "Electrolyzer Losses (kWh)",
            min_value=0.0,
            max_value=100.0,
            value=1.0,
            step=0.1
        )
        
        hour = st.slider(
            "Hour of Day",
            min_value=0,
            max_value=23,
            value=12,
            step=1
        )
    
    with col2:
        st.subheader("Prediction Result")
        
        # Create one-hot encoding for solar condition
        condition_one_hot = {f"SolarCond_{cond}": 1 if cond == selected_condition else 0 
                            for cond in condition_categories}
        
        # Create input for prediction
        input_data = {
            'Solar Irradiance (W/m^2)': irradiance,
            'Electrolyser Power (kW)': electrolyser_power,
            'Ambient Temperature (°C)': ambient_temp,
            'Electrolyzer Losses (kWh)': electrolyzer_losses,
            'Hour': hour,
        }
        
        # Add solar condition one-hot encoding
        input_data.update(condition_one_hot)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Scale numeric features
        input_df_scaled = input_df.copy()
        input_df_scaled[numeric_features] = scaler_enhanced.transform(input_df[numeric_features])
        
        # Make prediction
        if st.button("Predict PV Production"):
            prediction = rf_enhanced.predict(input_df_scaled)[0]
            
            # Show prediction with nice formatting
            st.markdown(
                f"""
                <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
                    <h3>Predicted PV Production</h3>
                    <h1 style="color: #1e88e5; font-size: 3em;">{prediction:.2f} kWh</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Get samples from similar conditions
            similar_samples = df[
                (df['Month'] == selected_month) &
                (df['Solar_Condition'] == selected_condition) &
                (df['Hour'] >= hour - 2) & 
                (df['Hour'] <= hour + 2)
            ]
            
            if not similar_samples.empty:
                avg_production = similar_samples['PV Production (kWh)'].mean()
                max_production = similar_samples['PV Production (kWh)'].max()
                min_production = similar_samples['PV Production (kWh)'].min()
                
                st.write(f"#### Similar Conditions Statistics")
                st.write(f"Found {len(similar_samples)} similar samples with these conditions")
                st.write(f"Average PV Production: {avg_production:.2f} kWh")
                st.write(f"Min PV Production: {min_production:.2f} kWh")
                st.write(f"Max PV Production: {max_production:.2f} kWh")
                
                # Distribution plot for similar conditions
                fig = px.histogram(
                    similar_samples, 
                    x='PV Production (kWh)',
                    nbins=20,
                    title=f'PV Production Distribution for Similar Conditions',
                    color_discrete_sequence=['lightblue']
                )
                
                # Add vertical line for prediction
                fig.add_vline(
                    x=prediction, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Prediction", 
                    annotation_position="top"
                )
                
                st.plotly_chart(fig)
            else:
                st.write("No similar samples found with these exact conditions.")
                
            # Show explanation of prediction factors
            st.write("#### Prediction Factors")
            st.write("""
            The prediction is based on the following key factors:
            1. Solar Irradiance - Higher irradiance generally leads to higher PV production
            2. Hour of Day - Peak production typically occurs around noon
            3. Solar Condition - Categories help the model understand different weather patterns
            4. Ambient Temperature - Can impact the efficiency of PV panels
            """)
