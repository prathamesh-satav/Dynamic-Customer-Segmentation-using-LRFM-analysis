# app.py

import streamlit as st
import pandas as pd
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax
from sklearn.metrics import silhouette_score
import plotly.express as px
from prophet import Prophet
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Segmentation App",
    layout="wide"
)

# --- Caching Data Loading and Preparation ---
@st.cache_data
def load_and_prepare_data():
    """
    Loads raw data, merges it, and engineers LRFM features.
    This function is cached to avoid reloading and recomputing on every interaction.
    """
    # Step 1: Load the Datasets
    try:
        transactions_df = pd.read_csv('Transactions_Cleaned.csv')
        demographics_df = pd.read_csv('CustomerDemographic_Cleaned.csv')
        address_df = pd.read_csv('CustomerAddress_Cleaned.csv')
    except FileNotFoundError:
        st.error("Error: Make sure all three CSV files are in the same folder as app.py.")
        return None

    # Step 2: Data Preparation
    master_df = pd.merge(transactions_df, demographics_df, on='customer_id', how='left')
    master_df = pd.merge(master_df, address_df, on='customer_id', how='left')
    master_df['transaction_date'] = pd.to_datetime(master_df['transaction_date'])
    abt = master_df[['customer_id', 'transaction_date', 'list_price']]
    abt = abt.rename(columns={'list_price': 'monetary_value'})
    abt.dropna(inplace=True)

    # Step 3: Time Series Aggregation
    abt_indexed = abt.set_index('transaction_date')
    time_series_df = abt_indexed.groupby('customer_id').resample('M').agg(
        frequency=('customer_id', 'size'),
        monetary=('monetary_value', 'sum')
    ).reset_index()
    time_series_df[['frequency', 'monetary']] = time_series_df[['frequency', 'monetary']].fillna(0)

    # Step 4: LRFM Feature Engineering
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
    
    return final_lrfm_df

# --- Main App UI ---
st.title("🛍️ Dynamic Customer Segmentation")
st.write("This application analyzes customer transaction data to identify distinct behavioral segments using the LRFM model and time series clustering. It also forecasts future spending for each segment.")

# Load the prepared data
final_lrfm_df = load_and_prepare_data()

if final_lrfm_df is not None:
    # --- Sidebar for User Inputs ---
    st.sidebar.header("⚙️ Analysis Settings")
    k_clusters = st.sidebar.slider("Select number of clusters (k):", min_value=2, max_value=8, value=4)
    run_button = st.sidebar.button("Run Analysis & Forecast")

    if run_button:
        # --- Step 5: Clustering ---
        st.header(f"Running Clustering for {k_clusters} Segments")
        with st.spinner('Reshaping and scaling data...'):
            features = ['length', 'recency', 'frequency', 'monetary']
            pivoted_df = final_lrfm_df.pivot(index='customer_id', columns='transaction_date', values=features)
            pivoted_df.fillna(0, inplace=True)

            n_customers = len(pivoted_df.index)
            n_timesteps = len(final_lrfm_df['transaction_date'].unique())
            n_features = len(features)
            data_array = pivoted_df.values.reshape(n_customers, n_timesteps, n_features)

            scaler = TimeSeriesScalerMinMax()
            scaled_data = scaler.fit_transform(data_array)
        st.success('Data scaling complete.')

        with st.spinner(f'Performing time series clustering with k={k_clusters}... This may take a moment.'):
            final_model = TimeSeriesKMeans(n_clusters=k_clusters, metric="dtw", max_iter=10, random_state=42, n_jobs=-1)
            final_labels = final_model.fit_predict(scaled_data)
            customer_clusters = pd.DataFrame({'customer_id': pivoted_df.index, 'cluster': final_labels})
        st.success('Clustering complete!')

        # --- Step 6: Analysis and Visualization ---
        st.header("📊 Cluster Analysis Results")
        analysis_df = pd.merge(final_lrfm_df, customer_clusters, on='customer_id')
        cluster_trends = analysis_df.groupby(['cluster', 'transaction_date'])['monetary'].mean().reset_index()

        # Plotly chart
        fig = px.line(
            cluster_trends, 
            x='transaction_date', 
            y='monetary', 
            color='cluster', 
            title='Average Monthly Spend per Customer Cluster',
            labels={'monetary': 'Average Monetary Value', 'transaction_date': 'Date'}
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary table
        st.write("#### Cluster Summary Statistics (Average LRFM values)")
        cluster_summary = analysis_df.groupby('cluster')[features].mean().round(2)
        cluster_summary['customer_count'] = analysis_df.groupby('cluster')['customer_id'].nunique()
        st.dataframe(cluster_summary)

        # --- Step 7: Forecasting ---
        st.header("📈 Future Spend Forecast")
        with st.spinner('Generating forecasts for each cluster...'):
            unique_clusters = analysis_df['cluster'].unique()
            unique_clusters.sort()

            for cluster_id in unique_clusters:
                cluster_data = cluster_trends[cluster_trends['cluster'] == cluster_id]
                
                prophet_df = cluster_data[['transaction_date', 'monetary']].rename(columns={
                    'transaction_date': 'ds',
                    'monetary': 'y'
                })
                
                model = Prophet()
                model.fit(prophet_df)
                
                future = model.make_future_dataframe(periods=6, freq='M')
                forecast = model.predict(future)
                
                # Plot using Matplotlib and display with st.pyplot
                fig_forecast = model.plot(forecast, xlabel='Date', ylabel='Average Spend')
                ax = fig_forecast.gca()
                ax.set_title(f'Forecast for Cluster {cluster_id}')
                st.pyplot(fig_forecast)
        st.success('Forecasting complete!')
    else:
        st.info("Adjust the settings in the sidebar and click 'Run Analysis & Forecast' to begin.")