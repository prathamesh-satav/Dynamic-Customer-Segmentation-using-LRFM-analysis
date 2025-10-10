# Dynamic Customer Segmentation & Forecasting

**Live Application Demo:** <https://dcsanalysis.streamlit.app/>
---

## 1. Project Overview

This project is an interactive data science application designed to perform dynamic customer segmentation. It addresses a common business problem: how to move beyond simple, static analysis and understand customers based on their behavior over time.

Inspired by the research paper *"A dynamic customer segmentation approach by combining LRFMS and multivariate time series clustering"*, this tool analyzes a raw transactional dataset to automatically group customers into distinct behavioral segments using an LRFM (Length, Recency, Frequency, Monetary) model.

The application identifies and characterizes key customer personas such as "Loyal Champions," "At-Risk Sleepers," and "Steady Supporters," providing a framework for data-driven, targeted marketing strategies. To make these insights actionable, it also includes a forecasting feature to predict the future spending trends for each segment.

This project was developed as a final year capstone, demonstrating a full data science workflow from data preparation and feature engineering to machine learning, visualization, and deployment as a live web application.

---

## 2. Key Features

-   **Interactive BI Dashboard:** Built with Streamlit to provide a user-friendly and exploratory experience similar to Power BI or Tableau.
-   **Dynamic Clustering:** Allows the user to select the number of customer segments (`k`) and the clustering algorithm (fast Euclidean vs. accurate DTW) via sidebar controls.
-   **Time-Series Analysis:** Models customer behavior on a month-by-month basis, capturing trends and patterns that static analysis would miss.
-   **LRFM Segmentation:** Utilizes the robust LRFM model to characterize customer value from multiple dimensions: Loyalty (Length), Engagement (Recency), Habit (Frequency), and Value (Monetary).
-   **Future-Facing Forecasts:** Uses the `Prophet` library to generate interactive 6-month spend forecasts for each segment, complete with uncertainty intervals.
-   **Deep-Dive Analysis:** Provides detailed demographic and geographic breakdowns (Age, Wealth, State) for each customer segment.

---

## 3. Technical Stack

-   **Language:** Python
-   **Core Libraries:**
    -   **Data Manipulation:** Pandas, NumPy
    -   **Time-Series Clustering:** `tslearn` (specifically `TimeSeriesKMeans`)
    -   **Forecasting:** `prophet` (by Facebook)
    -   **Machine Learning Utilities:** `scikit-learn`
-   **Dashboard & Visualization:**
    -   **Web Framework:** Streamlit
    -   **Interactive Charts:** Plotly (Radar, Pie, Bar, Scatter), Matplotlib
-   **Deployment:** Streamlit Community Cloud, GitHub

---

## 4. Methodology & Workflow

The project follows a standard data science pipeline:

1.  **Data Preparation & Cleaning:**
    -   The three raw datasets (`Transactions`, `Demographics`, `Address`) are loaded using Pandas.
    -   They are merged into a single Analytical Base Table (ABT).
    -   Data is cleaned by filtering for 'Approved' order statuses, converting data types (e.g., `transaction_date` to datetime), and handling missing values.

2.  **Time-Series Aggregation:**
    -   The transactional log is converted into a monthly time series for each customer using `pandas.resample('M')`.
    -   For each month, `Frequency` (count of purchases) and `Monetary` (sum of purchases) are calculated.

3.  **LRFM Feature Engineering:**
    -   The time-series data is enriched with the final LRFM features, calculated for each customer for each monthly time step:
        -   **Length (L):** Tenure of the customer in days, calculated from their very first purchase to their latest purchase within that period.
        -   **Recency (R):** Days since the customer's last purchase at the end of that period.
        -   **Frequency (F):** Number of transactions in the period.
        -   **Monetary (M):** Total amount spent in the period.

4.  **Multivariate Time-Series Clustering:**
    -   The data is reshaped into a 3D array (`n_customers`, `n_timesteps`, `n_features`) required by `tslearn`.
    -   All features are scaled to a `[0, 1]` range using `TimeSeriesScalerMinMax` to ensure they are weighted equally.
    -   `TimeSeriesKMeans` is used to group customers with similar LRFM patterns over time. The dashboard allows the user to choose between the fast `euclidean` metric and the more accurate but slower `dtw` (Dynamic Time Warping) metric.

5.  **Forecasting:**
    -   The historical average monthly spend for each identified cluster is aggregated.
    -   This aggregated time-series data is fed into a `Prophet` model for each segment.
    -   A 6-month forecast is generated, complete with `yhat_upper` and `yhat_lower` values, which are used to visualize the uncertainty interval.

---

## 5. How to Run This Project Locally

To run this application on your local machine, please follow these steps:

1.  **Prerequisites:**
    -   Python 3.8 - 3.10
    -   `git` for cloning the repository.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/prathamesh-satav/Dynamic-Customer-Segmentation-using-LRFM-analysis.git](https://github.com/prathamesh-satav/Dynamic-Customer-Segmentation-using-LRFM-analysis.git)
    cd Dynamic-Customer-Segmentation-using-LRFM-analysis
    ```

3.  **Create and Activate a Virtual Environment:**
    *This is highly recommended to avoid conflicts with other projects.*
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    python -m venv .venv
    .venv\Scripts\activate
    ```

4.  **Install the Required Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    The application should now be open and running in your web browser!

---

## 6. Project Structure
.
├── 📁 data/
│   ├── 📄 Transactions_Cleaned.csv
│   ├── 📄 CustomerDemographic_Cleaned.csv
│   └── 📄 CustomerAddress_Cleaned.csv
├── 📄 app.py
├── 📄 requirements.txt
└── 📄 README.md
