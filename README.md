# Dynamic Customer Segmentation & Forecasting

**Live App Demo: https://dcsanalytics.streamlit.app/**

## 1. Project Overview

This project is an intelligent customer segmentation tool built as a final year capstone project. It analyzes a transactional dataset to automatically group customers into distinct behavioral segments using an LRFM model and time series clustering.

The tool identifies key customer personas such as "Loyal Champions," "At-Risk Sleepers," and "New Potentials," and provides targeted marketing recommendations for each. It also includes a forecasting feature to predict the future spending trend for each segment.

## 2. Tech Stack

* **Language:** Python
* **Libraries:** Pandas, scikit-learn, tslearn, Prophet, Matplotlib
* **Dashboard:** Streamlit
* **Deployment:** Streamlit Community Cloud

## 3. How to Run This Project Locally

1.  **Clone the repository:**
    ```bash
    git clone [your-repo-url]
    cd [your-repo-name]
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 4. Key Results & Insights

The analysis identified four key customer segments with distinct behaviors. The most valuable segment, "Loyal Champions," consists of only 450 customers but contributes the highest revenue, while the largest segment, "Steady Supporters," forms the stable base of the business.



Strategic recommendations, such as VIP programs for top customers and re-engagement campaigns for at-risk ones, were developed based on these insights.
