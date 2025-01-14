import streamlit as st
import pandas as pd

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/Full Dataset with months.csv')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Month'] = df['DateTime'].dt.strftime('%B')
    df['Hour'] = df['DateTime'].dt.hour
    return df

# Initialize filters
def initialize_filters(dataframes):
    if 'selected_month' not in st.session_state:
        st.session_state.selected_month = None
    if 'hour_range' not in st.session_state:
        st.session_state.hour_range = (0, 23)

        # Get unique months from all dataframes
    all_months = set()
    for df in dataframes:
        if 'Month' in df.columns:
            all_months.update(df['Month'].unique())
    st.session_state.all_months = sorted(list(all_months))

# Display filters
def display_filters():
    st.sidebar.header('Filters')

    st.session_state.selected_month = st.sidebar.selectbox(
        'Select Month',
        options=st.session_state.all_months,
        key='month_filter'
    )

    st.session_state.hour_range = st.sidebar.slider(
        'Select Hour Range',
        min_value=0,
        max_value=23,
        value=st.session_state.hour_range,
        key='hour_range_filter'
    )

# Apply filters
def apply_filters(df):
    if 'Month' in df.columns and 'Hour' in df.columns:
        return df[
            (df['Month'] == st.session_state.selected_month) &
            (df['Hour'].between(st.session_state.hour_range[0], st.session_state.hour_range[1]))
        ]
    return df
