import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- Load and prepare data ---
@st.cache_data
def load_data():
    transactions_url = 'https://raw.githubusercontent.com/prathamesh-satav/Dynamic-Customer-Segmentation-using-LRFM-analysis/refs/heads/main/Transactions_Cleaned.csv' # Paste your raw URL here
    demographics_url = 'https://raw.githubusercontent.com/prathamesh-satav/Dynamic-Customer-Segmentation-using-LRFM-analysis/refs/heads/main/CustomerDemographic_Cleaned.csv' # Paste your raw URL here
    
    transactions_df = pd.read_csv(transactions_url)
    demographics_df = pd.read_csv(demographics_url)
    final_df = pd.merge(demographics_df, transactions_df, on='customer_id', how='left')
    final_df['transaction_date'] = pd.to_datetime(final_df['transaction_date'])
    return final_df

final_df = load_data()

# --- Calculate LRFM ---
def calculate_lrfm(df):
    snapshot_date = df['transaction_date'].max().date() + pd.Timedelta(days=1)
    rfm_df = df.groupby('customer_id').agg(
        Recency=('transaction_date', lambda x: (snapshot_date - x.max().date()).days),
        Frequency=('transaction_date', 'count'),
        Length=('transaction_date', lambda x: (x.max().date() - x.min().date()).days),
        Monetary=('Profit', 'sum')
    ).reset_index()
    return rfm_df

rfm_df = calculate_lrfm(final_df)
rfm_df = rfm_df.replace([np.inf, -np.inf], 0)

# --- Scale data for clustering ---
def scale_data(df):
    features = ['Recency', 'Frequency', 'Length', 'Monetary']
    df_log = df.copy()
    for col in features:
        df_log[col] = np.log1p(df_log[col])
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df_log[features]), columns=features, index=df_log.index)
    return scaled_df

scaled_df = scale_data(rfm_df)

# --- K-Means Clustering ---
def run_kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(df)
    return df

k_clusters = 4
rfm_df = run_kmeans(scaled_df.copy(), k_clusters)
rfm_df['customer_id'] = final_df['customer_id'].unique()

# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Dynamic Customer Segmentation Dashboard")

# Get average LRFM for each cluster
cluster_analysis = rfm_df.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Length': 'mean',
    'Monetary': 'mean'
}).reset_index()

# Interactive Plots
col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Segments by LRFM")
    fig = px.bar(cluster_analysis, x='Cluster', y='Recency', title='Recency by Cluster', color='Cluster')
    st.plotly_chart(fig)
with col2:
    st.subheader("Customer Demographics")
    fig = px.pie(final_df, names='gender', title='Gender Distribution')
    st.plotly_chart(fig)

# Display raw data for each cluster
st.subheader("Segment Data")
selected_cluster = st.selectbox("Select a Cluster to view data:", options=cluster_analysis['Cluster'].unique())
st.dataframe(rfm_df[rfm_df['Cluster'] == selected_cluster])
