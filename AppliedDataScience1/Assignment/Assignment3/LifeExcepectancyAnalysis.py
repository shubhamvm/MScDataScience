# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 13:41:45 2024

@author: Shubham Verma
Student Id: 22099668
email: sv23abk@herts.ac.uk
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import sklearn.preprocessing as pp
import matplotlib
from scipy.stats import gaussian_kde
import plotly.express as px
from plotly.subplots import make_subplots
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'

# student id for saving the plots
student_id = '22099668_'


def one_silhouette(xy, n):
    """
    Calculates silhouette score for n clusters.

    Parameters:
    - xy (DataFrame): Data for clustering.
    - n (int): Number of clusters.

    Returns:
    - float: Silhouette score.
    """
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)  # Fit done on x, y pairs
    labels = kmeans.labels_
    # Calculate the silhouette score
    score = skmet.silhouette_score(xy, labels)
    return score


def max_silhouette_score(df):
    """
    Finds the optimal number of clusters based on silhouette score.

    Parameters:
    - df (DataFrame): Data for clustering.

    Returns:
    - int: Optimal number of clusters.
    """
    minimum = scaler(df)
    # print('minimum: ', minimum)

    lSilhouetteScores = []
    # Calculate silhouette score for 2 to 10 clusters
    for ic in range(2, 11):
        score = one_silhouette(df, ic)
        print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
        lSilhouetteScores.append(score)

    # Get the index of the maximum value
    max_index = lSilhouetteScores.index(max(lSilhouetteScores)) + 2
    # print(max_index)
    return max_index


def map_corr(df, size=10):
    """
    Creates a heatmap of the correlation matrix for each pair of columns in the dataframe.

    Parameters:
    - df (DataFrame): Input data.
    - size (int): Vertical and horizontal size of the plot (in inches).

    Returns:
    - None
    """
    # Drop unnecessary columns
    columns_to_drop = ['Time', 'Time Code', 'Country Code', 'Country Name']
    df = df.drop(columns=columns_to_drop)
    # Remove rows with at least one NaN or empty value in-place
    df.dropna(inplace=True)
    # print(df.head(20))
    if df.isnull().any().any():
        print("Warning: NaN values detected in the DataFrame. Handle them before generating the heatmap.")
        return

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f", ax=ax)
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    # Save the plot as an image file
    plt.savefig(student_id + "Corr_matrix.png", dpi=300)

    plt.show()

    return None


def scaler(df):
    """
    Normalize columns of a DataFrame to the 0-1 range.

    Parameters:
    - df (DataFrame): Input data.

    Returns:
    - tuple: Normalized DataFrame, DataFrame of minimum values, DataFrame of maximum values.
    """
    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df - df_min) / (df_max - df_min)

    return df, df_min, df_max


def correlated_columns(df, correlation_threshold):
    """
    Selects columns with high correlation from a DataFrame.

    Parameters:
    - df (DataFrame): Input data.
    - correlation_threshold (float): Threshold for correlation.

    Returns:
    - DataFrame: Selected columns.
    """
    # Drop unnecessary columns
    columns_to_drop = ['Time', 'Time Code', 'Country Code', 'Country Name']
    df = df.drop(columns=columns_to_drop)

    # Remove rows with at least one NaN or empty value
    df = df.dropna()

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Find pairs of highly correlated columns
    highly_correlated_pairs = [(i, j) for i in range(len(corr_matrix.columns)) for j in range(
        i + 1, len(corr_matrix.columns)) if abs(corr_matrix.iloc[i, j]) > correlation_threshold]

    # Extract the column names to keep
    columns_to_keep = []
    for i, j in highly_correlated_pairs:
        columns_to_keep.append(corr_matrix.columns[i])
        columns_to_keep.append(corr_matrix.columns[j])

    # Create a new DataFrame with selected columns
    df_selected = df[columns_to_keep]
    df_selected = df_selected.loc[:, ~df_selected.columns.duplicated()]
    return df_selected


def ClustterAnalysis(df, cluster_count, lHeaders, lLabels):
    """
    Performs cluster analysis and generates a scatter plot.

    Parameters:
    - df (DataFrame): Input data.
    - cluster_count (int): Number of clusters.
    - lHeaders (list): List of column names for x and y axes.
    - lLabels (list): List of labels for x and y axes.

    Returns:
    - None
    """
    # Setup a scaler object
    scaler = pp.RobustScaler()
    scaler.fit(df)
    # And no to the scaling
    norm = scaler.transform(df)
    cm = matplotlib.colormaps["Paired"]
    # Set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=cluster_count, n_init=20)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(norm)  # Fit done on x, y pairs
    # Extract cluster labels
    labels = kmeans.labels_
    # Extract the estimated cluster centres and convert to original scales
    cen = kmeans.cluster_centers_
    cen = scaler.inverse_transform(cen)
    xkmeans = cen[:, 0]
    ykmeans = cen[:, 1]
    # Extract x and y values of data points
    x = df[lHeaders[0]]
    y = df[lHeaders[1]]
    plt.figure(figsize=(8.0, 8.0))
    # Plot data with kmeans cluster number
    plt.scatter(x, y, 10, labels, marker="o", cmap=cm)
    # Show cluster centres
    plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    plt.xlabel(lLabels[0])
    plt.ylabel(lLabels[1])

    # Save the plot as an image file
    plt.savefig(student_id + lHeaders[1] + "Clustter.png", dpi=300)

    plt.show()

    return None


def distribution_plot(df):
    """
    Generates distribution plots for each column in the DataFrame.

    Parameters:
    - df (DataFrame): Input data.

    Returns:
    - None
    """
    # Drop unnecessary columns
    columns_to_drop = ['Time', 'Time Code', 'Country Code', 'Country Name']
    df = df.drop(columns=columns_to_drop)
    # Remove rows with at least one NaN or empty value in-place
    df.dropna(inplace=True)

    # Calculate the number of rows and columns for the subplot grid
    num_cols = len(df.columns)
    num_rows = (num_cols - 1) // 3 + 1

    # Plot distribution for each column
    plt.figure(figsize=(15, 5 * num_rows))

    for i, column in enumerate(df.columns):
        # Adjust the number of columns as needed
        plt.subplot(num_rows, 3, i + 1)

        # Plot histogram
        plt.hist(df[column], bins=30, color='skyblue',
                 edgecolor='black', density=True)

        # Plot KDE line
        kde = gaussian_kde(df[column].dropna())
        x_vals = df[column].dropna().sort_values()
        plt.plot(x_vals, kde(x_vals), color='orange', label='KDE')

        plt.title(f'Distribution of {column}')
        plt.xlabel('Values')
        plt.ylabel('Density')
        plt.legend()

    # Save the plot as an image file
    plt.savefig(student_id + "distribution_plot.png", dpi=300)

    plt.show()

    return None


def regplot(df):
    """
    Generates a scatter plot with regression line and color bar.

    Parameters:
    - df (DataFrame): Input data.

    Returns:
    - None
    """
    x1 = "Adult Literacy rate"
    y1 = "Life expectancy"
    z1 = "Alcohol consumption"
    # Scatter plot with color bar
    points = plt.scatter(df[x1], df[y1], c=df[z1], s=20, cmap="Spectral")

    # Add a color bar
    plt.colorbar(points, label=z1)

    # Set limits
    plt.xlim(df[x1].min(), df[x1].max())
    plt.ylim(df[y1].min(), df[y1].max())

    # Build the plot
    sns.regplot(x=x1, y=y1, data=df, scatter=False, color=".1")

    # Add labels
    plt.xlabel(x1)
    plt.ylabel(y1)

    # Save the plot as an image file
    plt.savefig(student_id + "regplot.png", dpi=300)

    # Show the plot
    plt.show()

    return None


def mapplot(df):
    """
    Generates a choropleth map using Plotly Express.

    Parameters:
    - df (DataFrame): Input data.

    Returns:
    - None
    """
    # Assuming you have a column with country names or codes, replace 'CountryNameColumn' with the actual column name
    country_name_column = 'Country Code'

    # Create a choropleth map
    fig = px.choropleth(df,
                        locations=country_name_column,
                        color='Life expectancy',
                        color_continuous_scale='Spectral',
                        hover_name=country_name_column,
                        title='Life Expectancy worldwide',
                        )
    # Set tight layout with minimal margin
    fig.update_layout(geo=dict(fitbounds="locations",
                      visible=False), margin=dict(l=0, r=0, b=0, t=0))

    
    # Save the plot as a PNG file
    pio.write_image(fig, student_id + "mapplot.png")
    
    # Show the map
    fig.show()

    return None


def bubble_map_plot(df):
    """
    Generates a scatter plot on a map with overlapping bubbles using Plotly Express.

    Parameters:
    - df (DataFrame): Input data.

    Returns:
    - None
    """
    # Fill NaN values in 'Alcohol consumption' with 0
    df['Alcohol consumption'].fillna(0, inplace=True)

    # Assuming you have a column with country names or codes, replace 'CountryNameColumn' with the actual column name
    country_name_column = 'Country Code'

    # Create a scatter plot on a map with overlapping bubbles
    fig = px.scatter_geo(df,
                         locations=country_name_column,
                         size='Alcohol consumption',
                         color='Life expectancy',
                         projection="natural earth",
                         title='Scatter Plot on Map: Alcohol Consumption and Life Expectancy',
                         )
    
    # Save the plot as a PNG file
    pio.write_image(fig, student_id + "bubble_map_plot.png")
    
    # Show the map
    fig.show()

    return None


def life_expectancy_prediction(df):
    """
    Performs life expectancy prediction using linear regression and generates a plot.

    Parameters:
    - df (DataFrame): Input data.

    Returns:
    - None
    """
    # X is axis value (Year), y is y axis value (life expectancy for UK)
    X = df[['Time']]
    y = df['United Kingdom [GBR]']

    # Handle missing values
    X.dropna(inplace=True)
    y = y[X.index]
    df = df.transpose()
    # Add a constant to the predictor variable for statsmodels
    X = sm.add_constant(X)

    # Initialize and fit the linear regression model
    model = sm.OLS(y, X).fit()

    # Predict life expectancy for the years 2000 to 2040
    future_years = pd.DataFrame({'Time': range(1990, 2041)})
    future_years_with_const = sm.add_constant(future_years)
    life_expectancy_predictions = model.predict(future_years_with_const)

    # Compute confidence intervals
    prediction_interval = model.get_prediction(
        future_years_with_const).conf_int()
    df = df.transpose()
    # Plot the results
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df['Time'], y=df['United Kingdom [GBR]'], mode='markers', name='Actual Life Expectancy'))
    fig.add_trace(go.Scatter(x=future_years['Time'], y=life_expectancy_predictions, mode='markers', marker=dict(
        symbol='star'), name='Predicted Life Expectancy'))
    fig.add_trace(go.Scatter(x=future_years['Time'], y=prediction_interval[:, 0], mode='lines', line=dict(
        dash='dash'), name='Lower CI'))
    fig.add_trace(go.Scatter(x=future_years['Time'], y=prediction_interval[:, 1], mode='lines', line=dict(
        dash='dash'), name='Upper CI'))

    confidence_level = 0.95  # Specify confidence level
    fig.update_layout(title=f'Life Expectancy Prediction for the United Kingdom (Linear Regression) - {confidence_level * 100}% Confidence Interval',
                      xaxis_title='Year',
                      yaxis_title='Life Expectancy',
                      legend=dict(x=0, y=1, traceorder='normal'))
    
    # Save the plot as a PNG file
    pio.write_image(fig, student_id + "life_expectancy_prediction.png")

    fig.show()

    return None


# Load the dataset
url1 = 'https://raw.githubusercontent.com/shubhamvm/MScDataScience/main/AppliedDataScience1/Assignment/Assignment3/ClustterAnalysisData.csv'
df = pd.read_csv(url1)

# Load the dataset
url2 = "https://raw.githubusercontent.com/shubhamvm/MScDataScience/main/AppliedDataScience1/Assignment/Assignment3/LifeExpectancyPredictionUK.csv"
df2 = pd.read_csv(url2)

heatmap = map_corr(df)

# Create a pair plot for the entire DataFrame
sns.pairplot(
    df.drop(columns=['Time', 'Time Code', 'Country Code', 'Country Name']))

# Set a correlation threshold
corr_thresh = 0.6
df_selected = correlated_columns(df, corr_thresh)
print(df_selected.head())

# Cluster1 analysis
selection1 = ['Life expectancy', 'Adult Literacy rate']
df_new1 = df_selected[selection1].copy()
lLabel1 = ['Life expectancy (Years)', 'Adult Literacy rate (%)']

# data cleaning
df_new1 = df_new1.dropna()

cluster_count1 = max_silhouette_score(df_new1)
analysis1 = ClustterAnalysis(df_new1, cluster_count1, selection1, lLabel1)

# Cluster2 analysis
selection2 = ['CO2 emissions', 'GDP']
df_new2 = df_selected[selection2].copy()
lLabel2 = ['CO2 emissions (kt)', 'GDP (US$)']

# data cleaning
df_new2 = df_new2.dropna()

cluster_count2 = max_silhouette_score(df_new2)
analysis2 = ClustterAnalysis(df_new2, cluster_count2, selection2, lLabel2)

# Cluster3 analysis
selection3 = ['Life expectancy', 'GDP growth']
df_new3 = df[selection3].copy()
lLabel3 = ['Life expectancy (Years)', 'GDP Growth (%)']

# data cleaning
df_new3 = df_new3.dropna()

cluster_count3 = max_silhouette_score(df_new3)
analysis3 = ClustterAnalysis(df_new3, cluster_count3, selection3, lLabel3)

# Cluster4 analysis
selection4 = ['Life expectancy', 'Alcohol consumption']
df_new4 = df[selection4].copy()
lLabel4 = ['Life expectancy (Years)', 'Total alcohol consumption per capita']

# data cleaning
df_new4 = df_new4.dropna()

cluster_count4 = max_silhouette_score(df_new4)
analysis4 = ClustterAnalysis(df_new4, cluster_count4, selection4, lLabel4)

# Distribution of each column
distribution_plot(df)

# Regplot
regplot(df)

# map with life expectancy
mapplot(df)

# map life expectancy and alcohol consumption
bubble_map_plot(df)

# predicting life expectancy of the UK
life_expectancy_prediction(df2)
