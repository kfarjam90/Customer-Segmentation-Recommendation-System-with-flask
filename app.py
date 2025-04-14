from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly
import json
from io import BytesIO
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import linregress

app = Flask(__name__)

# Load and preprocess data
def load_data():
    df = pd.read_excel('Online Retail.xlsx')
    df = df.dropna(subset=['CustomerID', 'Description'])
    df.drop_duplicates(inplace=True)
    df = df[df['UnitPrice'] > 0]
    df.reset_index(drop=True, inplace=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def create_customer_features(df):
    df['Transaction_Status'] = np.where(df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed')
    df['InvoiceDay'] = df['InvoiceDate'].dt.date
    customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()
    most_recent_date = df['InvoiceDay'].max()
    customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
    most_recent_date = pd.to_datetime(most_recent_date)
    customer_data['Days_Since_Last_Purchase'] = (most_recent_date - customer_data['InvoiceDay']).dt.days
    customer_data.drop(columns=['InvoiceDay'], inplace=True)
    
    total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)
    
    total_products_purchased = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    total_products_purchased.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)
    
    customer_data = pd.merge(customer_data, total_transactions, on='CustomerID')
    customer_data = pd.merge(customer_data, total_products_purchased, on='CustomerID')
    
    df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
    total_spend = df.groupby('CustomerID')['Total_Spend'].sum().reset_index()
    
    average_transaction_value = total_spend.merge(total_transactions, on='CustomerID')
    average_transaction_value['Average_Transaction_Value'] = average_transaction_value['Total_Spend'] / average_transaction_value['Total_Transactions']
    
    customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
    customer_data = pd.merge(customer_data, average_transaction_value[['CustomerID', 'Average_Transaction_Value']], on='CustomerID')
    
    unique_products_purchased = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
    unique_products_purchased.rename(columns={'StockCode': 'Unique_Products_Purchased'}, inplace=True)
    
    customer_data = pd.merge(customer_data, unique_products_purchased, on='CustomerID')
    
    df['Day_Of_Week'] = df['InvoiceDate'].dt.dayofweek
    df['Hour'] = df['InvoiceDate'].dt.hour
    
    days_between_purchases = df.groupby('CustomerID')['InvoiceDay'].apply(lambda x: (x.diff().dropna()).apply(lambda y: y.days))
    average_days_between_purchases = days_between_purchases.groupby('CustomerID').mean().reset_index()
    average_days_between_purchases.rename(columns={'InvoiceDay': 'Average_Days_Between_Purchases'}, inplace=True)
    
    favorite_shopping_day = df.groupby(['CustomerID', 'Day_Of_Week']).size().reset_index(name='Count')
    favorite_shopping_day = favorite_shopping_day.loc[favorite_shopping_day.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Day_Of_Week']]
    
    favorite_shopping_hour = df.groupby(['CustomerID', 'Hour']).size().reset_index(name='Count')
    favorite_shopping_hour = favorite_shopping_hour.loc[favorite_shopping_hour.groupby('CustomerID')['Count'].idxmax()][['CustomerID', 'Hour']]
    
    customer_data = pd.merge(customer_data, average_days_between_purchases, on='CustomerID')
    customer_data = pd.merge(customer_data, favorite_shopping_day, on='CustomerID')
    customer_data = pd.merge(customer_data, favorite_shopping_hour, on='CustomerID')
    
    customer_country = df.groupby(['CustomerID', 'Country']).size().reset_index(name='Number_of_Transactions')
    customer_main_country = customer_country.sort_values('Number_of_Transactions', ascending=False).drop_duplicates('CustomerID')
    customer_main_country['Is_UK'] = customer_main_country['Country'].apply(lambda x: 1 if x == 'United Kingdom' else 0)
    customer_data = pd.merge(customer_data, customer_main_country[['CustomerID', 'Is_UK']], on='CustomerID', how='left')

    # Calculate the total number of transactions made by each customer
    total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()

    # Calculate the number of cancelled transactions for each customer
    cancelled_transactions = df[df['Transaction_Status'] == 'Cancelled']
    cancellation_frequency = cancelled_transactions.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    cancellation_frequency.rename(columns={'InvoiceNo': 'Cancellation_Frequency'}, inplace=True)

    customer_data = pd.merge(customer_data, cancellation_frequency, on='CustomerID', how='left')
    customer_data['Cancellation_Frequency'].fillna(0, inplace=True)
    customer_data['Cancellation_Rate'] = customer_data['Cancellation_Frequency'] / total_transactions['InvoiceNo']

    # Extract month and year from InvoiceDate
    df['Year'] = df['InvoiceDate'].dt.year
    df['Month'] = df['InvoiceDate'].dt.month

    # Calculate monthly spending for each customer
    monthly_spending = df.groupby(['CustomerID', 'Year', 'Month'])['Total_Spend'].sum().reset_index()

    # Calculate Seasonal Buying Patterns: We are using monthly frequency as a proxy for seasonal buying patterns
    seasonal_buying_patterns = monthly_spending.groupby('CustomerID')['Total_Spend'].agg(['mean', 'std']).reset_index()
    seasonal_buying_patterns.rename(columns={'mean': 'Monthly_Spending_Mean', 'std': 'Monthly_Spending_Std'}, inplace=True)

    # Replace NaN values in Monthly_Spending_Std with 0, implying no variability for customers with single transaction month
    seasonal_buying_patterns['Monthly_Spending_Std'].fillna(0, inplace=True)

    # Calculate Trends in Spending
    # We are using the slope of the linear trend line fitted to the customer's spending over time as an indicator of spending trends
    def calculate_trend(spend_data):
        # If there are more than one data points, we calculate the trend using linear regression
        if len(spend_data) > 1:
            x = np.arange(len(spend_data))
            slope, _, _, _, _ = linregress(x, spend_data)
            return slope
        # If there is only one data point, no trend can be calculated, hence we return 0
        else:
            return 0

    # Apply the calculate_trend function to find the spending trend for each customer
    spending_trends = monthly_spending.groupby('CustomerID')['Total_Spend'].apply(calculate_trend).reset_index()
    spending_trends.rename(columns={'Total_Spend': 'Spending_Trend'}, inplace=True)

    # Merge the new features into the customer_data dataframe
    customer_data = pd.merge(customer_data, seasonal_buying_patterns, on='CustomerID')
    customer_data = pd.merge(customer_data, spending_trends, on='CustomerID')

    # Changing the data type of 'CustomerID' to string as it is a unique identifier and not used in mathematical operations
    #customer_data['CustomerID'] = customer_data['CustomerID'].astype(str)
    # Convert data types of columns to optimal types
    customer_data = customer_data.convert_dtypes()

    model = IsolationForest(contamination=0.05, random_state=0)
    # Fitting the model on our dataset (converting DataFrame to NumPy to avoid warning)
    customer_data['Outlier_Scores'] = model.fit_predict(customer_data.iloc[:, 1:].to_numpy())

    # Creating a new column to identify outliers (1 for inliers and -1 for outliers)
    customer_data['Is_Outlier'] = [1 if x == -1 else 0 for x in customer_data['Outlier_Scores']]

    # Separate the outliers for analysis
    outliers_data = customer_data[customer_data['Is_Outlier'] == 1]

    # Remove the outliers from the main dataset
    customer_data_cleaned = customer_data[customer_data['Is_Outlier'] == 0]

    # Drop the 'Outlier_Scores' and 'Is_Outlier' columns
    customer_data_cleaned = customer_data_cleaned.drop(columns=['Outlier_Scores', 'Is_Outlier'])

    # Reset the index of the cleaned data
    customer_data_cleaned.reset_index(drop=True, inplace=True)    
        
    return customer_data_cleaned

def perform_clustering(customer_data):
    customer_data_cleaned = customer_data.copy()
    
    columns_to_exclude = ['CustomerID', 'Is_UK', 'Day_Of_Week']
    columns_to_scale = customer_data_cleaned.columns.difference(columns_to_exclude)
    scaler = StandardScaler()
    customer_data_scaled = customer_data_cleaned.copy()
    customer_data_scaled[columns_to_scale] = scaler.fit_transform(customer_data_scaled[columns_to_scale])
    
    customer_data_scaled.set_index('CustomerID', inplace=True)
    pca = PCA(n_components=6)
    customer_data_pca = pca.fit_transform(customer_data_scaled)
    customer_data_pca = pd.DataFrame(customer_data_pca, columns=['PC'+str(i+1) for i in range(pca.n_components_)])
    customer_data_pca.index = customer_data_scaled.index
    
    kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=100, random_state=0)
    kmeans.fit(customer_data_pca)
    
    cluster_frequencies = Counter(kmeans.labels_)
    label_mapping = {label: new_label for new_label, (label, _) in
                 enumerate(cluster_frequencies.most_common())}
    label_mapping = {v: k for k, v in {2: 1, 1: 0, 0: 2}.items()}
    new_labels = np.array([label_mapping[label] for label in kmeans.labels_])
    
    customer_data_cleaned['cluster'] = new_labels
    customer_data_pca['cluster'] = new_labels
    
    return customer_data_cleaned, customer_data_pca, pca

# Load data and perform clustering at startup
df = load_data()
customer_data = create_customer_features(df)
customer_data_cleaned, customer_data_pca, pca = perform_clustering(customer_data)

@app.route('/')
def index():
    # Basic stats
    total_customers = len(customer_data_cleaned)
    total_transactions = len(df)
    
    # Cluster distribution
    cluster_percentage = (customer_data_pca['cluster'].value_counts(normalize=True) * 100).reset_index()
    cluster_percentage.columns = ['Cluster', 'Percentage']
    cluster_percentage.sort_values(by='Cluster', inplace=True)
    
    # Convert to list of dicts for template
    cluster_data = cluster_percentage.to_dict('records')
    
    return render_template('index.html', 
                         total_customers=total_customers,
                         total_transactions=total_transactions,
                         cluster_data=cluster_data)

@app.route('/cluster_distribution')
def cluster_distribution():
    colors = ['#e8000b', '#1ac938', '#023eff']
    cluster_percentage = (customer_data_pca['cluster'].value_counts(normalize=True) * 100).reset_index()
    cluster_percentage.columns = ['Cluster', 'Percentage']
    cluster_percentage.sort_values(by='Cluster', inplace=True)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Percentage', y='Cluster', data=cluster_percentage, orient='h', palette=colors)
    for index, value in enumerate(cluster_percentage['Percentage']):
        ax.text(value+0.5, index, f'{value:.2f}%')
    ax.set_title('Distribution of Customers Across Clusters', fontsize=14)
    ax.set_xticks(np.arange(0, 50, 5))
    ax.set_xlabel('Percentage (%)')
    
    # Save plot to bytes
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('cluster_distribution.html', plot_url=plot_url)

@app.route('/cluster_3d')
def cluster_3d():
    colors = ['#e8000b', '#1ac938', '#023eff']
    
    # Create separate data frames for each cluster
    cluster_0 = customer_data_pca[customer_data_pca['cluster'] == 0]
    cluster_1 = customer_data_pca[customer_data_pca['cluster'] == 1]
    cluster_2 = customer_data_pca[customer_data_pca['cluster'] == 2]
    
    # Create a 3D scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=cluster_0['PC1'], y=cluster_0['PC2'], z=cluster_0['PC3'],
                               mode='markers', marker=dict(color=colors[0], size=5, opacity=0.4), name='Cluster 0'))
    fig.add_trace(go.Scatter3d(x=cluster_1['PC1'], y=cluster_1['PC2'], z=cluster_1['PC3'],
                               mode='markers', marker=dict(color=colors[1], size=5, opacity=0.4), name='Cluster 1'))
    fig.add_trace(go.Scatter3d(x=cluster_2['PC1'], y=cluster_2['PC2'], z=cluster_2['PC3'],
                               mode='markers', marker=dict(color=colors[2], size=5, opacity=0.4), name='Cluster 2'))
    
    fig.update_layout(
        title=dict(text='3D Visualization of Customer Clusters in PCA Space', x=0.5),
        scene=dict(
            xaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC1'),
            yaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC2'),
            zaxis=dict(backgroundcolor="#fcf0dc", gridcolor='white', title='PC3'),
        ),
        width=900,
        height=800
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('cluster_3d.html', graphJSON=graphJSON)

@app.route('/cluster_characteristics')
def cluster_characteristics():
    colors = ['#e8000b', '#1ac938', '#023eff']
    
    # Standardize the data
    df_customer = customer_data_cleaned.set_index('CustomerID')
    scaler = StandardScaler()
    df_customer_standardized = scaler.fit_transform(df_customer.drop(columns=['cluster'], axis=1))
    df_customer_standardized = pd.DataFrame(df_customer_standardized, 
                                          columns=df_customer.columns[:-1], 
                                          index=df_customer.index)
    df_customer_standardized['cluster'] = df_customer['cluster']
    
    # Calculate centroids
    cluster_centroids = df_customer_standardized.groupby('cluster').mean()
    
    # Create radar chart
    labels = np.array(cluster_centroids.columns)
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    labels = np.concatenate((labels, [labels[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(polar=True), nrows=1, ncols=3)
    
    for i, color in enumerate(colors):
        data = cluster_centroids.loc[i].tolist()
        data += data[:1]
        ax[i].fill(angles, data, color=color, alpha=0.4)
        ax[i].plot(angles, data, color=color, linewidth=2, linestyle='solid')
        ax[i].set_title(f'Cluster {i}', size=20, color=color, y=1.1)
        ax[i].set_xticks(angles[:-1])
        ax[i].set_xticklabels(labels[:-1])
        ax[i].grid(color='grey', linewidth=0.5)
    
    # Save plot to bytes
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return render_template('cluster_characteristics.html', plot_url=plot_url)

@app.route('/customer_lookup')
def customer_lookup():
    customer_ids = customer_data_cleaned['CustomerID'].unique().tolist()
    return render_template('customer_lookup.html', customer_ids=customer_ids)

@app.route('/get_customer_data', methods=['POST'])
def get_customer_data():
    customer_id = float(request.form['customer_id'])
    
    customer_info = customer_data_cleaned[customer_data_cleaned['CustomerID'] == customer_id].iloc[0]
    customer_transactions = df[df['CustomerID'] == customer_id].sort_values('InvoiceDate', ascending=False).head(10)
    
    # Convert to dict for JSON response
    response = {
        'cluster': int(customer_info['cluster']),
        'days_since_last_purchase': int(customer_info['Days_Since_Last_Purchase']),
        'total_transactions': int(customer_info['Total_Transactions']),
        'total_products_purchased': int(customer_info['Total_Products_Purchased']),
        'total_spend': float(customer_info['Total_Spend']),
        'avg_transaction_value': float(customer_info['Average_Transaction_Value']),
        'unique_products_purchased': int(customer_info['Unique_Products_Purchased']),
        'transactions': customer_transactions.to_dict('records')
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)