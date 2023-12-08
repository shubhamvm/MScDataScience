# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 21:48:11 2023

@author: Shubham Verma
Student ID: 22099668
email: sv23abk@herts.ac.uk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import describe


def extract_population_growth_and_transpose(df):
    """
    Extract the data for population growth and return two dataframes
    df_PopGrowth: years as columns
    df_PopGrowth_T: countries as columns (transpose)

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.

    Returns: df_PopGrowth, df_PopGrowth_T
    """
    columns_to_extract = ['Country Name', 'Country Code', 'Indicator Name',
                          'Indicator Code'] + [str(year) for year in range(1990, 2017)]
    df_PopGrowth = df[df['Indicator Code']
                      == 'SP.POP.GROW'][columns_to_extract]
    l_countries_growth = ['LIC', 'MIC', 'HIC']
    df_PopGrowth = df_PopGrowth[df_PopGrowth['Country Code'].isin(
        l_countries_growth)]
    df_PopGrowth = df_PopGrowth[['Country Name', '1996',
                                 '1999', '2002', '2005', '2008', '2011', '2014']]

    df_PopGrowth.set_index('Country Name', inplace=True)

    # Transpose the DataFrame using transpose() method
    df_PopGrowth_T = df_PopGrowth.transpose()

    return df_PopGrowth, df_PopGrowth_T


def line_plot(dataframe, headers, legends, labels):
    """
    Create a line plot to visualize data series specified by 'headers' over the x-axis.

    Args:
    dataframe (pandas.DataFrame): The DataFrame containing the data.
    headers (list): A list containing the x-axis and headers of the columns to plot.
    legends (list): A list containing the legends for the data series being plotted.
    labels (list): A list containing the labels for the x-axis, y-axis, and title.

    Plots the multiple lines plot as per given data and instructions

    Returns:
    None
    """
    # Create a new figure for the plot
    plt.figure()

    # Plot the data series specified by 'headers' with years on the x-axis
    for head in headers[1:]:  # Exclude the x-axis from the list
        plt.plot(dataframe[headers[0]], dataframe[head])

    # Add labels and title to the plot
    plt.xlabel(labels[0], fontweight='bold')  # Label the x-axis
    plt.ylabel(labels[1], fontweight='bold')  # Label the y-axis
    plt.title(labels[2])  # Set the plot title

    # Rotate y-axis tick labels for better visibility
    plt.yticks(rotation=45)

    # Add a legend to identify the data series
    plt.legend(legends)

    # Remove white space on the left and right of the plot by setting the x-axis limits
    plt.xlim(min(dataframe[headers[0]]), dataframe[headers[0]].max())
    current_values = plt.gca().get_yticks()
    # using format string '{:.0f}' here but you can choose others
    plt.gca().set_yticklabels(
        ['{:.0f} billion'.format(x/1000000000) for x in current_values])
    
    # Save the plot as an image file
    plt.savefig("LineChart_22099668.png", dpi=300)
    
    # Display the plot
    plt.show()

    return


def pie_chart(dataframe, labels, title):
    """
    Create a pie chart to visualize data using custom autopct and labels.

    Args:
    dataframe (pandas.DataFrame): The dataframe or data series containing the relevant data.
    labels (list): A list of labels to be used for the pie chart segments.
    title (str): A string of title to be used for the pie chart.

    Returns:
    None
    """
    plt.figure()

    myexplode = [0, 0.2, 0]

    # Create the pie chart with custom autopct and font size for labels
    plt.pie(dataframe, labels=labels, autopct='%.2f%%',
            explode=myexplode, textprops={'fontsize': 9}, shadow=True)
    

    # Set the title for the pie chart
    plt.title(title)
    
    # Save the pie chart as an image file
    plt.savefig("PieChart_22099668.png", dpi=300)
    
    # Display the pie chart
    plt.show()

    return


def bar_plot(categories, bar1_percentage, bar2_percentage, bar_labels, labels):
    """
    Create a bar plot with dashed line to visualize two data series over time.

    Args:
    categories (list): The DataFrame containing the data.
    bar1_percentage (list): The name of the column for the x-axis (Years in this case).
    bar2_percentage (list): The name of the first data series to plot.
    bar_labels (list): The name of the second data series to plot (Inflation in this case).
    labels (list): Label for the x-axis.

    and plots the bars chart as per given data and instructions
    Returns: None
    """
    # Set up the bar positions
    bar_width = 0.35
    bar_positions1 = range(len(categories))
    bar_positions2 = [pos + bar_width for pos in bar_positions1]

    # Plot the bars
    bars1 = plt.bar(bar_positions1, bar1_percentage,
                    width=bar_width, label=bar_labels[0])
    bars2 = plt.bar(bar_positions2, bar2_percentage,
                    width=bar_width, label=bar_labels[1])

    # Add labels inside the bars
    for bar in bars1:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_val,
                 f'{y_val}%', ha='center', va='bottom')

    for bar in bars2:
        y_val = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, y_val,
                 f'{y_val}%', ha='center', va='bottom')

    # Customize the chart
    plt.title(labels[0])
    plt.xlabel(labels[1], fontweight='bold')
    plt.ylabel(labels[2], fontweight='bold')
    plt.xticks([pos + bar_width / 2 for pos in bar_positions1], categories)
    plt.legend()

    # Set y-axis limits to 0 and 100
    plt.ylim(0, 100)
    
    # Save the bar chart as an image file
    plt.savefig("BarChart_22099668.png", dpi=300)
    
    # Show the plot
    plt.show()

    return


def area_chart(df_pop_growth_T, headers, labels):
    """
    Create an Area plot with visualize two data series over time.

    Args:
    df_pop_growth_T (pandas.DataFrame): The DataFrame containing the data.
    headers (list): list of data to plot.
    labels (list): Label for the title, x-axis, and y-axis.

    and plots the area chart as per given data and instructions
    Returns: None
    """
    # Create the area chart
    plt.fill_between(df_pop_growth_T.index,
                     df_pop_growth_T[headers[0]], alpha=0.3)
    plt.fill_between(df_pop_growth_T.index,
                     df_pop_growth_T[headers[1]], alpha=0.3)
    plt.fill_between(df_pop_growth_T.index,
                     df_pop_growth_T[headers[2]], alpha=0.3)

    # Add labels and title
    plt.xlabel(labels[1], fontweight='bold')
    plt.ylabel(labels[2], fontweight='bold')
    plt.title(labels[0])

    # Show legend
    plt.legend(headers)
    
    # Save the area chart as an image file
    plt.savefig("AreaChart_22099668.png", dpi=300)
    
    # Display the plot
    plt.show()

    return


url1 = 'https://raw.githubusercontent.com/shubhamvm/MScDataScience/main/' 
url1 += 'AppliedDataScience1/Assignment/Assignment%202/'
url1 += 'API_19_DS2_en_csv_v2_5998250.csv'

df = pd.read_csv(url1)

# Extract relevant columns by the countries from 1990 to 2015
columns_to_extract = ['Country Name', 'Country Code', 'Indicator Name',
                      'Indicator Code'] + [str(year) for year in range(1990, 2017)]
df_TotalPop = df[df['Indicator Code'] == 'SP.POP.TOTL'].dropna()[
    columns_to_extract]
df_UrbPopulation = df[df['Indicator Code'] == 'SP.URB.TOTL'].dropna()[
    columns_to_extract]
df_PopGrowth, df_PopGrowth_T = extract_population_growth_and_transpose(df)

# 3 statistics methods
# Calculate the correlation matrix using numpy
correlation_matrix_np = np.corrcoef(df_PopGrowth_T.values, rowvar=False)
print("Correlation Matrix (Numpy):")
print(correlation_matrix_np)

# Calculate the covariance matrix using numpy
covariance_matrix_np = np.cov(df_PopGrowth_T.values, rowvar=False)
print("Covariance Matrix (Numpy):")
print(covariance_matrix_np)

# Use scipy to get summary statistics
summary_scipy = describe(df_PopGrowth_T, axis=1)
print('Summary of population growth (Scipy):')
print(summary_scipy)

df_PopGrowth.style.background_gradient(cmap='Blues')

headers_area_chart = ['High income', 'Middle income', 'Low income']
labels_area_chart = ['Area chart of population % and income category',
                     'Years', '% of growth']
area_chart(df_PopGrowth_T, headers_area_chart, labels_area_chart)

# Define the headers, legends, and labels for the first line plot
headers_line_plot = ['Country Name', '1996', '2006', '2016']
legends_line_plot = ['1996', '2006', '2016']
labels_line_plot = ['Income Category', 'Total Population',
                    'Income category vs Total population worldwide']
df_IncomePopulation = df_TotalPop[df_TotalPop['Country Code'].isin(
    ['LIC', 'MIC', 'HIC'])]
line_plot(df_IncomePopulation, headers_line_plot,
          legends_line_plot, labels_line_plot)

# Extracting data for further plots
df_UrbanPop = df[df['Indicator Code'] ==
                 'SP.URB.TOTL.IN.ZS'][columns_to_extract]
df_UrbanInc = df_UrbanPop[df_UrbanPop['Country Code'].isin(
    ['LIC', 'MIC', 'HIC'])]
df_UrbanInc = df_UrbanInc[['Country Name', '1996', '2006', '2016']]

HighUrb = df_UrbanInc[(df_UrbanInc['Country Name'] ==
                       'High income')]['2016'].to_string(index=False)
HighNonUrb = 100 - int(float(HighUrb))
HighTot = df_IncomePopulation[(
    df_IncomePopulation['Country Name'] == 'High income')]['2016'].to_string(index=False)
MidUrb = df_UrbanInc[(df_UrbanInc['Country Name'] ==
                      'Middle income')]['2016'].to_string(index=False)
MidNonUrb = 100 - int(float(MidUrb))
MidTot = df_IncomePopulation[(
    df_IncomePopulation['Country Name'] == 'Middle income')]['2016'].to_string(index=False)
LowUrb = df_UrbanInc[(df_UrbanInc['Country Name'] ==
                      'Low income')]['2016'].to_string(index=False)
LowNonUrb = 100 - int(float(LowUrb))
LowTot = df_IncomePopulation[(
    df_IncomePopulation['Country Name'] == 'Low income')]['2016'].to_string(index=False)

values_pie_chart = [int(float(HighTot)), int(
    float(MidTot)), int(float(LowTot))]
title_pie_chart = 'Population in 2016 incomewise'
labels_pie_chart = ['High income', 'Middle income', 'Low income']

# Call a function to create a pie chart with the provided data and labels
pie_chart(values_pie_chart, labels_pie_chart, title_pie_chart)

categories_bar_plot = ['High Income', 'Medium Income', 'Low Income']
bar1_percentage_bar_plot = [int(float(HighUrb)), int(
    float(MidUrb)), int(float(LowUrb))]
bar2_percentage_bar_plot = [HighNonUrb, MidNonUrb, LowNonUrb]
labels_bar = ['Urban population(%)', 'Non urban population(%)']
labels_bar_plot = ['Comparison of urban population with income categories in 2016',
                   'Categories', 'Percentage']

bar_plot(categories_bar_plot, bar1_percentage_bar_plot,
         bar2_percentage_bar_plot, labels_bar, labels_bar_plot)
