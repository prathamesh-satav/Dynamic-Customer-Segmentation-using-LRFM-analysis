# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

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
        # UPDATED: File paths reverted to the main directory
        transactions_df = pd.read_csv('Transactions_Cleaned.csv')
        demographics_df = pd.read_csv('CustomerDemographic_Cleaned.csv')
        address_df = pd.read_csv('CustomerAddress_Cleaned.csv')
    except FileNotFoundError:
        st.error("Error: Make sure all three CSV files are in the SAME folder as app.py.")
        return None, None, None

    master_df = pd.merge(transactions_df, demographics_df, on='customer_id', how='left')
    master_df = pd.merge(master_df, address_df, on='customer_id', how='left')
    master_df['transaction_date'] = pd.to_datetime(master_df['transaction_date'])
    
    master_df = master_df[master_df['order_status'] == 'Approved'].copy()

    abt = master_df[['customer_id', 'transaction_date', 'list_price', 'gender', 'wealth_segment', 'state', 'Age']].copy()
    abt = abt.rename(columns={'list_price': 'monetary_value'})
    abt.dropna(subset=['customer_id', 'transaction_date', 'monetary_value'], inplace=True)
    abt['customer_id'] = abt['customer_id'].astype(int)

    abt_indexed = abt.set_index('transaction_date')
    time_series_df = abt_indexed.groupby('customer_id').resample('M').agg(
        frequency=('customer_id', 'size'),
        monetary=('monetary_value', 'sum')
    ).reset_index()
    time_series_df[['frequency', 'monetary']] = time_series_df[['frequency', 'monetary']].fillna(0)

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
    
    return final_lrfm_df, abt, (transactions_df, demographics_df, address_df)

@st.cache_data
def run_clustering(lrfm_df, k, metric):
    """Performs time-series clustering on the LRFM data with a chosen metric."""
    st.info(f"Running clustering with k={k} and metric='{metric}'. Please wait...")
    
    features = ['length', 'recency', 'frequency', 'monetary']
    pivoted_df = lrfm_df.pivot(index='customer_id', columns='transaction_date', values=features)
    pivoted_df.fillna(0, inplace=True)

    data_array = pivoted_df.values.reshape(len(pivoted_df.index), -1, len(features))
    scaler = TimeSeriesScalerMinMax()
    scaled_data = scaler.fit_transform(data_array)

    model = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=5, random_state=42, n_jobs=-1)
    labels = model.fit_predict(scaled_data)
    
    customer_clusters = pd.DataFrame({'customer_id': pivoted_df.index, 'cluster': labels})
    return customer_clusters

# --- Main App ---

st.title("Dynamic Customer Segmentation Dashboard")
st.markdown("An interactive tool to explore customer behavior over time, identify key segments, and forecast future trends.")

# Load data
lrfm_data, demographic_data, raw_data = load_and_prepare_data()

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

    # --- Run Clustering (main computation) ---
    customer_clusters = run_clustering(lrfm_data, k_clusters, metric)
    analysis_df = pd.merge(demographic_data, customer_clusters, on='customer_id')
    
    # --- NEW LAYOUT: SINGLE PAGE ---
    
    # --- SECTION 1: ABOUT THE DATA ---
    st.markdown("---")
    st.header("About the Source Data")
    st.markdown("""
    This application is powered by three separate datasets, which are cleaned and merged to create the analytical model. The data covers transactions and customer information for the year 2017. Below is a small sample from each source file to provide context.
    """)
    
    st.subheader("1. Transactions Dataset")
    st.markdown("This file contains individual transaction records, including the customer ID, the date of the transaction, and the price of the items purchased. This is the core data used to calculate customer spending habits.")
    st.dataframe(raw_data[0].head())

    st.subheader("2. Customer Demographics Dataset")
    st.markdown("This file contains demographic information for each customer, such as their gender, age, and assigned wealth segment. This data is used to enrich our understanding of the final customer groups.")
    st.dataframe(raw_data[1].head())

    st.subheader("3. Customer Address Dataset")
    st.markdown("This file contains address information for each customer, which allows us to analyze the geographic distribution of our customer segments.")
    st.dataframe(raw_data[2].head())

    # --- SECTION 2: THE DASHBOARD ---
    st.markdown("---")
    st.header("Interactive Dashboard")

    # --- Overall Business Performance KPIs ---
    st.subheader("Overall Business Performance")
    st.markdown("These Key Performance Indicators (KPIs) provide a high-level snapshot of the business's health based on the entire dataset. They help us understand the overall scale of revenue, the size of the customer base, and the average value derived from transactions.")
    total_revenue = analysis_df['monetary_value'].sum()
    total_customers = analysis_df['customer_id'].nunique()
    total_transactions = len(demographic_data)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Revenue", f"${total_revenue:,.0f}", help="The total value of all approved transactions.")
        col2.metric("Total Customers", f"{total_customers:,}", help="The number of unique customers who made at least one purchase.")
        col3.metric("Avg. Revenue / Customer", f"${total_revenue/total_customers:,.2f}", help="Also known as ARPC, this is the average value each customer brings to the business.")
        col4.metric("Avg. Order Value", f"${total_revenue/total_transactions:,.2f}", help="Also known as AOV, this is the average value of a single transaction.")
    
    st.markdown("---")

    # --- Segment Deep-Dive ---
    st.subheader("Customer Segment Deep-Dive")
    st.markdown("Here, we explore the distinct customer groups (personas) that our machine learning model has identified. By selecting a persona from the dropdown menu, you can see detailed information about that group's size, value, and unique characteristics.")
    
    segment_lrfm_avg = lrfm_data.groupby('customer_id').mean().reset_index()
    segment_lrfm_avg = pd.merge(segment_lrfm_avg, customer_clusters, on='customer_id')
    
    cluster_personas = {
        0: "🌟 Loyal Champions", 1: "⏳ At-Risk Sleepers", 2: "👍 Steady Supporters",
        3: "🌱 New Potentials", 4: "High-Value Occasional", 5: "Low-Value Churning",
        6: "Engaged but Low-Spend", 7: "High-Potential Newbies"
    }
    customer_clusters['persona'] = customer_clusters['cluster'].map(cluster_personas)
    analysis_df = pd.merge(analysis_df, customer_clusters[['customer_id', 'persona']], on='customer_id')

    persona_options = sorted(customer_clusters['persona'].unique())
    selected_persona = st.selectbox("Choose a customer persona to analyze:", options=persona_options)
    
    segment_data = analysis_df[analysis_df['persona'] == selected_persona]
    
    original_cluster_num = next((key for key, value in cluster_personas.items() if value == selected_persona), 0)
    segment_lrfm_data = segment_lrfm_avg[segment_lrfm_avg['cluster'] == original_cluster_num]

    st.write(f"#### Profile of: {selected_persona}")
    seg_customers = segment_data['customer_id'].nunique()
    seg_revenue = segment_data['monetary_value'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Number of Customers", f"{seg_customers:,}")
    c2.metric("% of Total Customers", f"{(seg_customers/total_customers)*100:.1f}%" if total_customers > 0 else "0%")
    c3.metric("Total Revenue from Segment", f"${seg_revenue:,.0f}")
    c4.metric("% of Total Revenue", f"{(seg_revenue/total_revenue)*100:.1f}%" if total_revenue > 0 else "0%")

    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Behavioral Profile (LRFM)")
        st.markdown("This radar chart compares the selected segment's behavior to the average customer across all four LRFM metrics. A **larger shape** indicates a more valuable and engaged segment, while the shape's bias towards a certain metric reveals the group's defining characteristic.")
        if not segment_lrfm_data.empty:
            lrfm_avg_all = lrfm_data.mean(numeric_only=True)
            lrfm_avg_segment = segment_lrfm_data.mean(numeric_only=True)
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=lrfm_avg_segment.values, theta=lrfm_avg_segment.index, fill='toself', name='Selected Segment'))
            fig.add_trace(go.Scatterpolar(r=lrfm_avg_all.values, theta=lrfm_avg_all.index, fill='toself', name='Overall Average'))
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No data for behavioral profile.")

    with col2:
        st.write("##### Wealth Distribution")
        st.markdown("This chart breaks down the wealth segments (e.g., Mass Customer, Affluent) within the chosen persona. This is crucial for tailoring marketing messages and product recommendations appropriately.")
        if not segment_data.empty:
            wealth_dist = segment_data['wealth_segment'].value_counts()
            fig = px.pie(wealth_dist, values=wealth_dist.values, names=wealth_dist.index, hole=.4)
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("No data for wealth distribution.")

    st.markdown("---")
    
    # --- Forecasting ---
    st.subheader("Future Spend Forecast")
    st.markdown("This section predicts the average monthly spend for each customer segment for the next 6 months. This allows the business to be proactive and plan for future revenue trends.")
    
    cluster_trends = lrfm_data.groupby(['customer_id', 'transaction_date'])['monetary'].sum().reset_index()
    cluster_trends = pd.merge(cluster_trends, customer_clusters, on='customer_id')
    cluster_trends = cluster_trends.groupby(['persona', 'transaction_date'])['monetary'].mean().reset_index()

    personas_to_forecast = sorted(cluster_trends['persona'].unique())
    forecast_options = ["All Segments"] + personas_to_forecast
    selected_forecast = st.selectbox("Select a segment to forecast:", options=forecast_options)

    all_forecasts = []
    for persona in personas_to_forecast:
        prophet_df = cluster_trends[cluster_trends['persona'] == persona][['transaction_date', 'monetary']].rename(columns={'transaction_date': 'ds', 'monetary': 'y'})
        if len(prophet_df) > 1:
            model = Prophet()
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=6, freq='M')
            forecast = model.predict(future)
            forecast['persona'] = persona
            all_forecasts.append(forecast)

    if all_forecasts:
        combined_forecast_df = pd.concat(all_forecasts)
        
        if selected_forecast == "All Segments":
            personas_to_plot = personas_to_forecast
            chart_title = "6-Month Spend Forecast for All Segments"
        else:
            personas_to_plot = [selected_forecast]
            chart_title = f"6-Month Spend Forecast for {selected_forecast}"

        fig_forecast = go.Figure()
        for persona in personas_to_plot:
            persona_df = combined_forecast_df[combined_forecast_df['persona'] == persona]
            fig_forecast.add_trace(go.Scatter(x=persona_df['ds'], y=persona_df['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=persona_df['ds'], y=persona_df['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.3)', name='Uncertainty'))
            fig_forecast.add_trace(go.Scatter(x=persona_df['ds'], y=persona_df['yhat'], mode='lines', name=f'Forecast: {persona}'))
            hist_df = cluster_trends[cluster_trends['persona'] == persona]
            fig_forecast.add_trace(go.Scatter(x=hist_df['transaction_date'], y=hist_df['monetary'], mode='markers', name=f'Historical: {persona}'))
        
        fig_forecast.update_layout(title=chart_title, xaxis_title="Date", yaxis_title="Average Monthly Spend", height=500)
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        st.markdown("""
        **How to read this chart?**
        - **Markers** represent the actual, historical average spending for the group.
        - The **solid colored line** is the model's prediction for future spending.
        - The **light shaded area** represents the "uncertainty interval" – this is the most likely range (best-case to worst-case) where future spending will fall.
        """)
    else: st.warning("Not enough data to generate forecasts.")

    # --- SECTION 3: METHODOLOGY ---
    st.markdown("---")
    st.header("Project Methodology")
    st.markdown("""
    This project uses an advanced, time-series approach to customer segmentation. Here is a breakdown of the core concepts that power this dashboard.
    """)

    st.subheader("1. What is Dynamic Segmentation?")
    st.markdown("""
    Traditional segmentation takes a single snapshot of a customer (like a photograph). It answers the question, "Who are our best customers right now?"
    
    **Dynamic Segmentation**, which this project uses, analyzes a customer's behavior over time (like a movie). It answers the question, "How are our customers' behaviors evolving?" This allows us to identify important trends and trajectories, such as a loyal customer who is becoming less active, which a static model would completely miss.
    """)

    st.subheader("2. The LRFM Model")
    st.markdown("""
    To create the "movie" for each customer, we build a "behavioral fingerprint" by calculating four key metrics every month:
    - **(L) Length:** How long has this person been a customer? This measures their long-term *loyalty*.
    - **(R) Recency:** How recently did they make a purchase? This measures their current *engagement*.
    - **(F) Frequency:** How often do they buy? This measures their purchasing *habit*.
    - **(M) Monetary:** How much money do they spend? This measures their direct *value*.
    """)

    st.subheader("3. Time-Series Clustering Algorithm")
    st.markdown("""
    We use a machine learning algorithm called **TimeSeries K-Means**. It is designed specifically to compare the year-long LRFM "fingerprints" of all customers and automatically group those with the most similar patterns. 
    
    You can choose between two methods in the sidebar for how the algorithm measures "similarity":
    - **Euclidean:** A very fast, standard calculation that compares data points directly.
    - **Dynamic Time Warping (DTW):** A more advanced and accurate method that can find similar patterns even if they are shifted in time (e.g., it can identify two customers who both buy seasonally, even if their seasons are a few months apart).
    """)

    st.subheader("4. The Forecasting Model")
    st.markdown("""
    To predict future trends, we use **Prophet**, a powerful and robust forecasting library developed by Facebook. It analyzes the historical spending trend of each segment to predict its likely spending for the next 6 months. A key feature is its ability to provide an "uncertainty interval," which gives us a realistic best-case and worst-case range for our forecast.
    """)

