# Dynamic Customer Segmentation & Forecasting

**Live Application Demo: https://dcsanalytics.streamlit.app/** 

![App Screenshot](https://drive.google.com/file/d/1ZiNrWx0sU3hVmhBVj4f5Nf7CfUiLwtmx/view?usp=sharing)

---

## 1. Project Overview

This project is an interactive data science application designed to perform dynamic customer segmentation. It addresses a common business problem: how to move beyond simple, static analysis and understand customers based on their behavior over time.

Inspired by the research paper *"A dynamic customer segmentation approach by combining LRFMS and multivariate time series clustering"*, this tool analyzes a raw transactional dataset to automatically group customers into distinct behavioral segments using an LRFM (Length, Recency, Frequency, Monetary) model.

The application identifies and characterizes key customer personas such as "Loyal Champions," "At-Risk Sleepers," and "Steady Supporters," providing a framework for data-driven, targeted marketing strategies. To make these insights actionable, it also includes a forecasting feature to predict the future spending trends for each segment.

This project was developed as a final year capstone, demonstrating a full data science workflow from data preparation and feature engineering to machine learning, visualization, and deployment as a live web application.

---

## 2. Key Features

* **Interactive Dashboard:** Built with Streamlit for a user-friendly and interactive experience.
* **Dynamic Clustering:** Allows the user to select the number of customer segments (`k`) to create.
* **Time Series Analysis:** Models customer behavior on a month-by-month basis instead of a single snapshot.
* **LRFM Segmentation:** Utilizes the robust LRFM model to characterize customer value from multiple dimensions.
* **Future-Facing Forecasts:** Uses the Prophet library to predict the average monthly spend for each segment for the next 6 months.
* **Actionable Insights:** Translates complex data into clear customer personas and strategic business recommendations.

---

## 3. Technical Stack

* **Language:** Python
* **Core Libraries:**
    * **Data Manipulation:** Pandas, NumPy
    * **Time Series Clustering:** `tslearn` (specifically `TimeSeriesKMeans` with DTW metric)
    * **Forecasting:** `prophet` (by Facebook)
    * **Machine Learning Utilities:** `scikit-learn`
* **Dashboard & Visualization:**
    * **Web Framework:** Streamlit
    * **Interactive Charts:** Plotly
    * **Static Plots:** Matplotlib
* **Deployment:** Streamlit Community Cloud, GitHub

---

## 4. Methodology & Workflow

The project follows a standard data science pipeline:

1.  **Data Preparation & Cleaning:**
    * The three raw datasets (`Transactions`, `Demographics`, `Address`) are loaded.
    * They are merged into a single Analytical Base Table (ABT).
    * Data types are corrected (e.g., `transaction_date` to datetime) and missing values are handled.

2.  **Time Series Aggregation:**
    * The transactional log is converted into a monthly time series for each customer.
    * For each month, `Frequency` (count of purchases) and `Monetary` (sum of purchases) are calculated.

3.  **LRFM Feature Engineering:**
    * The time series data is enriched with the final LRFM features:
        * **Length (L):** Tenure of the customer in days, calculated from their first purchase to their latest purchase in the period.
        * **Recency (R'):** Days since the customer's last purchase within the period.
        * **Frequency (F):** Number of transactions in the period.
        * **Monetary (M):** Total amount spent in the period.

4.  **Multivariate Time Series Clustering:**
    * The data is reshaped into a 3D array (`n_customers`, `n_timesteps`, `n_features`) required by `tslearn`.
    * Features are scaled to a `[0, 1]` range to ensure equal weighting.
    * `TimeSeriesKMeans` is used with the **Dynamic Time Warping (DTW)** metric to group customers with similar LRFM patterns over time.

5.  **Forecasting:**
    * The historical average monthly spend for each identified cluster is fed into a `Prophet` model.
    * A 6-month forecast is generated, complete with uncertainty intervals, for each segment.

---

## 5. How to Run This Project Locally

To run this application on your local machine, please follow these steps:

1.  **Prerequisites:**
    * Python 3.8 - 3.10
    * `git` for cloning the repository.

2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
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
    *Ensure your three data files (`Transactions_Cleaned.csv`, etc.) are in the root directory.*
    ```bash
    streamlit run app.py
    ```
    The application should now be open and running in your web browser!

---

## 6. Project Structure
├── 📄 app.py                     # The main Streamlit application script
├── 📄 requirements.txt           # A list of all necessary Python libraries
├── 📄 README.md                  # This detailed project explanation
└── 📁 data/                      # Folder containing the source data
    ├── 📄 Transactions_Cleaned.csv
    ├── 📄 CustomerDemographic_Cleaned.csv
    └── 📄 CustomerAddress_Cleaned.csv
