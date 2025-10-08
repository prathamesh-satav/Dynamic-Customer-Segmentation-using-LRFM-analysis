# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions (Cached for performance) ---

@st.cache_data
def load_and_prepare_data():
    """Loads, merges, and prepares the LRFM time-series data."""
    try:
        transactions_df = pd.read_csv('Transactions_Cleaned.csv')
        demographics_df = pd.read_csv('CustomerDemographic_Cleaned.csv')
        address_df = pd.read_csv('CustomerAddress_Cleaned.csv')
    except FileNotFoundError:
        st.error("Error: Make sure all three CSV files are in the same folder as app.py.")
        return None, None

    # Merge and create analytical base table
    master_df = pd.merge(transactions_df, demographics_df, on='customer_id', how='left')
    master_df = pd.merge(master_df, address_df, on='customer_id', how='left')
    master_df['transaction_date'] = pd.to_datetime(master_df['transaction_date'])
    
    # Filter for approved transactions only
    master_df = master_df[master_df['order_status'] == 'Approved'].copy()

    abt = master_df[['customer_id', 'transaction_date', 'list_price', 'gender', 'wealth_segment', 'state', 'Age']].copy()
    abt = abt.rename(columns={'list_price': 'monetary_value'})
    abt.dropna(subset=['customer_id', 'transaction_date', 'monetary_value'], inplace=True)
    abt['customer_id'] = abt['customer_id'].astype(int)

    # Time Series Aggregation
    abt_indexed = abt.set_index('transaction_date')
    time_series_df = abt_indexed.groupby('customer_id').resample('M').agg(
        frequency=('customer_id', 'size'),
        monetary=('monetary_value', 'sum')
    ).reset_index()
    time_series_df[['frequency', 'monetary']] = time_series_df[['frequency', 'monetary']].fillna(0)

    # LRFM Feature Engineering
    first_purchase = abt.groupby('customer_id')['transaction_date'].min().reset_index()
    first_purchase.rename(columns={'transaction_date': 'first_purchase_date'}, inplace=True)

    abt['transaction_month'] = abt['transaction_date'].dt.to_period('M').dt.start_time
    last_purchase_in_period = abt.groupby(['customer_id', 'transaction_month'])['transaction_date'].max().reset_index()
    last_purchase_in_period.rename(columns={'transaction_date': 'last_purchase_date', 'transaction_month': 'transaction_date'}, inplace=True)

    ts_final = pd.merge(time_series_df, first_purchase, on='customer_id', how='left')
    ts_final = pd.merge(ts_final, last_purchase_in_period, on=['customer_id', 'transaction_date'], how='left')
    ts_final['last_purchase_date'] = ts_final.groupby('customer_id')['last_purchase_date'].ffill()
    ts_final['length'] = (ts_final['last_purchase_date'] - ts_final['first_purchase_date']).dt.days
    ts_final['recency'] = (ts_final['transaction_date'].dt.to_period('M').dt.end_time - ts_final['last_purchase_date']).dt.days
    ts_final.fillna(0, inplace=True)
    
    features = ['length', 'recency', 'frequency', 'monetary']
    final_lrfm_df = ts_final[['customer_id', 'transaction_date'] + features]
    
    return final_lrfm_df, abt

@st.cache_data
def run_clustering(lrfm_df, k, metric):
    """Performs time-series clustering on the LRFM data with a chosen metric."""
    st.info(f"Running clustering with k={k} and metric='{metric}'. Please wait...")
    
    features = ['length', 'recency', 'frequency', 'monetary']
    pivoted_df = lrfm_df.pivot(index='customer_id', columns='transaction_date', values=features)
    pivoted_df.fillna(0, inplace=True)

    n_customers = len(pivoted_df.index)
    n_timesteps = len(lrfm_df['transaction_date'].unique())
    n_features = len(features)
    data_array = pivoted_df.values.reshape(n_customers, n_timesteps, n_features)

    scaler = TimeSeriesScalerMinMax()
    scaled_data = scaler.fit_transform(data_array)

    model = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=5, random_state=42, n_jobs=-1)
    labels = model.fit_predict(scaled_data)
    
    customer_clusters = pd.DataFrame({'customer_id': pivoted_df.index, 'cluster': labels})
    return customer_clusters

# --- Main App ---

st.title("📊 Customer Segmentation Dashboard")
st.markdown("An interactive dashboard to explore dynamic customer segments and forecast future trends.")

# Load data
lrfm_data, demographic_data = load_and_prepare_data()

if lrfm_data is not None:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("Dashboard Settings")
    
    k_clusters = st.sidebar.slider("1. Select number of clusters (k):", min_value=2, max_value=8, value=4,
                                   help="Choose how many customer segments you want to identify.")
    
    metric_choice = st.sidebar.radio(
        "2. Choose a clustering algorithm:",
        ('Euclidean (Fast)', 'DTW (Accurate, but slow)'),
        help="Euclidean is very fast. DTW is more accurate for time-series patterns but can take several minutes to run."
    )
    
    metric = 'euclidean' if metric_choice == 'Euclidean (Fast)' else 'dtw'

    # --- Clustering ---
    customer_clusters = run_clustering(lrfm_data, k_clusters, metric)
    analysis_df = pd.merge(demographic_data, customer_clusters, on='customer_id')

    # --- Overall Business KPIs ---
    st.header("📈 Overall Business Performance")
    total_revenue = analysis_df['monetary_value'].sum()
    total_customers = analysis_df['customer_id'].nunique()
    total_transactions = len(demographic_data) # Using original ABT for transaction count

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Customers", f"{total_customers:,}")
    col3.metric("Avg. Revenue / Customer", f"${total_revenue/total_customers:,.2f}")
    col4.metric("Avg. Order Value", f"${total_revenue/total_transactions:,.2f}")

    st.markdown("---")

    # --- Segment Deep-Dive ---
    st.header("👥 Customer Segment Deep-Dive")
    
    # Add a selector for the segment
    segment_lrfm_avg = lrfm_data.groupby('customer_id').mean().reset_index()
    segment_lrfm_avg = pd.merge(segment_lrfm_avg, customer_clusters, on='customer_id')
    
    # Rename clusters for better understanding
    cluster_personas = {
        0: "🌟 Loyal Champions",
        1: "⏳ At-Risk Sleepers",
        2: "👍 Steady Supporters",
        3: "🌱 New Potentials",
        4: "High-Value Occasional",
        5: "Low-Value Churning",
        6: "Engaged but Low-Spend",
        7: "High-Potential Newbies"
    }
    customer_clusters['persona'] = customer_clusters['cluster'].map(cluster_personas)
    analysis_df = pd.merge(analysis_df, customer_clusters[['customer_id', 'persona']], on='customer_id')

    # Handle case where selected persona might not exist after re-clustering
    persona_options = sorted(customer_clusters['persona'].unique())
    selected_persona = st.selectbox("Choose a customer persona to analyze:", options=persona_options)
    
    segment_data = analysis_df[analysis_df['persona'] == selected_persona]
    
    # Find the original cluster number for the selected persona name
    original_cluster_num = next(key for key, value in cluster_personas.items() if value == selected_persona)
    segment_lrfm_data = segment_lrfm_avg[segment_lrfm_avg['cluster'] == original_cluster_num]


    # --- KPIs for the Selected Segment ---
    st.subheader(f"Profile of: {selected_persona}")
    seg_customers = segment_data['customer_id'].nunique()
    seg_revenue = segment_data['monetary_value'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of Customers", f"{seg_customers:,}")
    c2.metric("% of Total Customers", f"{(seg_customers/total_customers)*100:.1f}%" if total_customers > 0 else "0%")
    c3.metric("Total Revenue from Segment", f"${seg_revenue:,.0f}")
    c4.metric("% of Total Revenue", f"{(seg_revenue/total_revenue)*100:.1f}%" if total_revenue > 0 else "0%")

    # --- Visualizations for the Selected Segment ---
    col1, col2 = st.columns(2)

    with col1:
        # LRFM Radar Chart
        st.write("#### Behavioral Profile (LRFM)")
        if not segment_lrfm_data.empty:
            lrfm_avg_all = lrfm_data.mean(numeric_only=True)
            lrfm_avg_segment = segment_lrfm_data.mean(numeric_only=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                  r=lrfm_avg_segment.values,
                  theta=lrfm_avg_segment.index,
                  fill='toself',
                  name='Selected Segment'
            ))
            fig.add_trace(go.Scatterpolar(
                  r=lrfm_avg_all.values,
                  theta=lrfm_avg_all.index,
                  fill='toself',
                  name='Overall Average'
            ))
            fig.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  range=[0, lrfm_data.max(numeric_only=True).max()]
                )),
              showlegend=True,
              height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for this segment to create a behavioral profile.")

    with col2:
        # Wealth Segment Donut Chart
        st.write("#### Wealth Distribution")
        if not segment_data.empty:
            wealth_dist = segment_data['wealth_segment'].value_counts()
            fig = px.pie(wealth_dist, values=wealth_dist.values, names=wealth_dist.index, hole=.4)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for wealth distribution.")
    
    col3, col4 = st.columns(2)
    with col3:
        # Age Distribution
        st.write("#### Age Distribution")
        if not segment_data.empty and 'Age' in segment_data.columns:
            age_bins = [0, 25, 35, 45, 55, 65, 100]
            age_labels = ['<25', '25-35', '36-45', '46-55', '56-65', '65+']
            segment_data['age_group'] = pd.cut(segment_data['Age'], bins=age_bins, labels=age_labels, right=False)
            age_dist = segment_data['age_group'].value_counts().sort_index()
            fig = px.bar(age_dist, x=age_dist.index, y=age_dist.values, labels={'x': 'Age Group', 'y': 'Number of Customers'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No age data available.")

    with col4:
        # Geographic Distribution
        st.write("#### Top 5 States")
        if not segment_data.empty and 'state' in segment_data.columns:
            state_dist = segment_data['state'].value_counts().nlargest(5)
            fig = px.bar(state_dist, x=state_dist.index, y=state_dist.values, labels={'x': 'State', 'y': 'Number of Customers'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No state data available.")

    st.markdown("---")

    # --- Forecasting Section ---
    st.header("🔮 Future Spend Forecast")
    st.write("Predicting the average monthly spend for each customer segment for the next 6 months.")

    # Calculate historical trends for forecasting
    cluster_trends = lrfm_data.groupby(['customer_id', 'transaction_date'])['monetary'].sum().reset_index()
    cluster_trends = pd.merge(cluster_trends, customer_clusters, on='customer_id')
    cluster_trends = cluster_trends.groupby(['persona', 'transaction_date'])['monetary'].mean().reset_index()

    # Get unique personas to forecast for
    personas_to_forecast = sorted(cluster_trends['persona'].unique())

    # ADDED: Dropdown for forecast selection
    forecast_options = ["All Clusters"] + personas_to_forecast
    selected_forecast = st.selectbox("Select a segment to forecast:", options=forecast_options)

    # Generate all forecasts
    all_forecasts = []
    for persona in personas_to_forecast:
        prophet_df = cluster_trends[cluster_trends['persona'] == persona][['transaction_date', 'monetary']].rename(columns={
            'transaction_date': 'ds',
            'monetary': 'y'
        })
        
        if len(prophet_df) > 1: # Prophet needs at least 2 data points
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future)
            forecast['persona'] = persona
            all_forecasts.append(forecast)

    if all_forecasts:
        combined_forecast_df = pd.concat(all_forecasts)

        # ADDED: Filter which personas to plot based on dropdown
        if selected_forecast == "All Clusters":
            personas_to_plot = personas_to_forecast
            chart_title = "6-Month Spend Forecast for All Customer Segments"
        else:
            personas_to_plot = [selected_forecast]
            chart_title = f"6-Month Spend Forecast for {selected_forecast}"

        # Create interactive Plotly figure
        fig_forecast = go.Figure()

        # MODIFIED: Loop over the filtered list of personas
        for persona in personas_to_plot:
            persona_df = combined_forecast_df[combined_forecast_df['persona'] == persona]
            # Add the uncertainty band (yhat_lower, yhat_upper)
            fig_forecast.add_trace(go.Scatter(
                x=persona_df['ds'],
                y=persona_df['yhat_upper'],
                mode='lines',
                line=dict(width=0),
                fillcolor='rgba(173, 216, 230, 0.3)',
                showlegend=False
            ))
            fig_forecast.add_trace(go.Scatter(
                x=persona_df['ds'],
                y=persona_df['yhat_lower'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(173, 216, 230, 0.3)',
                name='Uncertainty'
            ))
            # Add the forecast line (yhat)
            fig_forecast.add_trace(go.Scatter(
                x=persona_df['ds'],
                y=persona_df['yhat'],
                mode='lines',
                name=f'Forecast: {persona}'
            ))
            # Add historical data points
            hist_df = cluster_trends[cluster_trends['persona'] == persona]
            fig_forecast.add_trace(go.Scatter(
                x=hist_df['transaction_date'],
                y=hist_df['monetary'],
                mode='markers',
                name=f'Historical: {persona}'
            ))
            
        fig_forecast.update_layout(
            title=chart_title, # MODIFIED: Use dynamic title
            xaxis_title="Date",
            yaxis_title="Average Monthly Spend",
            height=600
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.warning("Not enough data to generate forecasts.")

