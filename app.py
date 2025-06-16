import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# Streamlit Page Config
st.set_page_config(page_title="FP Shop Data Analysis", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        .main {background-color: #f8f9fa;}
        .stTextInput, .stFileUploader, .stSelectbox {border-radius: 10px;}
        h1 {color: #2E86C1; text-align: center;}
        h2 {color: #117A65;}
        h3 {color: #D35400;}
        .stAlert {border-radius: 10px;}
        .css-1aumxhk {background-color: white !important; border-radius: 10px;}
        .ml-section {background-color: #E8F8F5; padding: 15px; border-radius: 10px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("Fair Price Shop Data Analysis")
st.markdown("### Analyze and visualize Fair Price Shop transactions dynamically.")

# File Upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset Loaded Successfully!")

    # Tabs
    tabs = ["Overview", "Top 10 Shops", "Transactions by District", "Commodity Distribution", "Utilization Rate", "Top 5 Districts", "Commodity by Shop"]
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(tabs)

    # Overview Tab
    with tab1:
        st.subheader("Overview of FP Shop Data")
        st.markdown("This section provides a *summary of the dataset*, including general statistics and insights.")
        st.write(df.describe())

    # Top 10 Shops by Total Amount
    with tab2:
        st.subheader("Top 10 Shops by Total Amount")
        st.markdown("This graph shows the *top 10 shops based on total sales*, helping identify high-performing shops.")
        top_shops = df.groupby("shopNo")["totalAmount"].sum().nlargest(10)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_shops.index, y=top_shops.values, palette="viridis")
        plt.xlabel("Shop Number")
        plt.ylabel("Total Amount")
        plt.title("Top 10 Shops by Total Amount")
        st.pyplot(plt)

    # Distribution of Transactions by District
    with tab3:
        st.subheader("Distribution of Transactions by District")
        st.markdown("This graph shows the *distribution of transactions across districts*, highlighting areas with the highest transaction activity.")
        plt.figure(figsize=(12, 6))
        sns.histplot(df["distName"], bins=30, kde=True, color="blue")
        plt.xticks(rotation=90)
        plt.xlabel("District Name")
        plt.ylabel("Number of Transactions")
        plt.title("Transactions by District")
        st.pyplot(plt)

    # Commodity Distribution
    with tab4:
        st.subheader("Commodity Distribution by District")
        st.markdown("This graph *illustrates how different commodities* are distributed across various districts.")
        district_data = df.groupby("distName")[['riceAfsc', 'riceFsc', 'riceAap', 'wheat', 'sugar', 'rgdal', 'kerosene', 'salt']].sum()
        fig, ax = plt.subplots(figsize=(12, 6))
        district_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        ax.set_ylabel("Quantity Distributed")
        ax.set_title("Commodity Distribution by District")
        st.pyplot(fig)

    # Utilization Rate
    with tab5:
        st.subheader("Ration Card Utilization Rate")
        st.markdown("This *heatmap shows the utilization rate* of ration cards in different districts, indicating active use of facilities.")
        df['utilization_rate'] = (df['noOfTrans'] / df['noOfRcs']) * 100
        utilization_pivot = df.pivot_table(values='utilization_rate', index='distName')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(utilization_pivot, cmap='coolwarm', annot=True, ax=ax)
        ax.set_title("Utilization Rate per District")
        st.pyplot(fig)

    # Top 5 Districts by Total Transactions
    with tab6:
        st.subheader("Top 5 Districts by Total Transactions")
        st.markdown("This graph shows the *top 5 districts with the highest number of transactions*, highlighting areas with the most activity.")
        top_districts = df.groupby("distName")["noOfTrans"].sum().nlargest(5)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_districts.index, y=top_districts.values, palette="magma")
        plt.xlabel("District Name")
        plt.ylabel("Total Transactions")
        plt.title("Top 5 Districts by Total Transactions")
        st.pyplot(plt)

    # Commodity Distribution by Shop
    with tab7:
        st.subheader("Commodity Distribution by Shop")
        st.markdown("This graph shows the *distribution of commodities across shops*, providing insights into shop-level allocation.")
        shop_data = df.groupby("shopNo")[['riceAfsc', 'riceFsc', 'riceAap', 'wheat', 'sugar', 'rgdal', 'kerosene', 'salt']].sum().nlargest(10, columns=['riceAfsc'])
        fig, ax = plt.subplots(figsize=(12, 6))
        shop_data.plot(kind='bar', stacked=True, ax=ax, colormap='plasma')
        ax.set_ylabel("Quantity Distributed")
        ax.set_title("Commodity Distribution by Shop")
        st.pyplot(fig)

    # Machine Learning Section
    st.markdown("---")
    st.markdown("## Machine Learning Insights")
    st.markdown("This section provides advanced insights using machine learning models.")

    # ML Tabs
    ml_tabs = ["Anomaly Detection", "District Clustering", "Resource Allocation Optimization"]
    tab8, tab9, tab10 = st.tabs(ml_tabs)

    # Anomaly Detection
    with tab8:
        st.subheader("Anomaly Detection in Transactions")
        st.markdown("This section identifies *unusual transaction patterns* that may indicate fraud or errors.")

        # Select features for anomaly detection
        features = ['noOfTrans', 'totalAmount', 'noOfRcs']
        X = df[features]

        # Train Isolation Forest model
        iso_forest = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination as needed
        df['anomaly'] = iso_forest.fit_predict(X)
        df['anomaly'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

        # Display results
        st.write("### Anomaly Detection Results")
        st.dataframe(df[['shopNo', 'distName', 'noOfTrans', 'totalAmount', 'noOfRcs', 'anomaly']])

        # Filter to show only anomalies
        if st.checkbox("Show Only Anomalies"):
            st.write("### Anomalies Only")
            anomalies_df = df[df['anomaly'] == 'Anomaly']
            st.dataframe(anomalies_df[['shopNo', 'distName', 'noOfTrans', 'totalAmount', 'noOfRcs', 'anomaly']])

        # Visualize anomalies
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='noOfTrans', y='totalAmount', hue='anomaly', data=df, palette='coolwarm')
        plt.title("Anomalies in Transactions")
        st.pyplot(plt)

    # District Clustering
    with tab9:
        st.subheader("District Clustering")
        st.markdown("This section groups districts into clusters based on transaction volume and resource allocation.")

        # Select features for clustering
        features = ['noOfTrans', 'totalAmount', 'noOfRcs', 'riceAfsc', 'riceFsc', 'wheat', 'sugar']
        X = df[features]

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Allow user to select number of clusters
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=5, value=3)

        # Train K-Means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        # Calculate Silhouette Score
        silhouette_avg = silhouette_score(X_scaled, df['cluster'])
        st.write(f"### Silhouette Score: {silhouette_avg:.2f}")
        st.markdown("The Silhouette Score ranges from -1 to 1, where a higher value indicates better-defined clusters.")

        # Display results
        st.write("### District Clustering Results")
        st.dataframe(df[['distName', 'noOfTrans', 'totalAmount', 'noOfRcs', 'cluster']])

        # Filter to show specific clusters
        cluster_filter = st.selectbox("Filter by Cluster", options=df['cluster'].unique())
        filtered_df = df[df['cluster'] == cluster_filter]
        st.write(f"### Districts in Cluster {cluster_filter}")
        st.dataframe(filtered_df[['distName', 'noOfTrans', 'totalAmount', 'noOfRcs', 'cluster']])

        # Visualize clusters
        plt.figure(figsize=(12, 6))
        sns.scatterplot(x='noOfTrans', y='totalAmount', hue='cluster', data=df, palette='viridis')
        plt.title("District Clusters")
        st.pyplot(plt)

    # Resource Allocation Optimization
    with tab10:
        st.subheader("Resource Allocation Optimization")
        st.markdown("This section predicts the optimal allocation of commodities based on demand patterns.")

        # Select features and target
        features = ['noOfRcs', 'noOfTrans']
        target = ['riceAfsc', 'riceFsc', 'wheat', 'sugar']
        X = df[features]
        y = df[target]

        # Add polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)

        # Train Linear Regression model
        model = LinearRegression()
        model.fit(X_poly, y)
        y_pred = model.predict(X_poly)

        # Calculate R² Score and MSE
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        st.write(f"### R² Score: {r2:.2f}")
        st.write(f"### Mean Squared Error (MSE): {mse:.2f}")

        # Display results
        st.write("### Predicted Commodity Requirements")
        df[['pred_riceAfsc', 'pred_riceFsc', 'pred_wheat', 'pred_sugar']] = y_pred
        st.dataframe(df[['distName', 'noOfRcs', 'noOfTrans', 'pred_riceAfsc', 'pred_riceFsc', 'pred_wheat', 'pred_sugar']])

        # Visualize predictions
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='noOfTrans', y='pred_riceAfsc', data=df, label='Predicted Rice (AFSC)')
        sns.lineplot(x='noOfTrans', y='pred_riceFsc', data=df, label='Predicted Rice (FSC)')
        plt.xlabel("Number of Transactions")
        plt.ylabel("Predicted Quantity")
        plt.title("Predicted Commodity Requirements")
        st.pyplot(plt)

else:
    st.sidebar.warning("⚠ Please upload a CSV file to proceed.")