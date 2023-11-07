#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 7 12:43:07 2023

@author: Shubham Verma
"""

# Importing necessary packages and libraries

import pandas as pd  # Import Pandas for data manipulation
import matplotlib.pyplot as plt  # Import Matplotlib for plotting


# defining user defined function linePlot to line plot
def line_plot(dataframe, lHeaders, lLegends, lLabels):
    """
    Create a line plot to visualize data series specified by 'lHeaders'
    over the x-axis.

    Args:
    dataframe (pandas.DataFrame): 
        The DataFrame containing the data.
    lHeaders (list): 
        A list containing the x-axis and headers of the columns to plot.
    lLegends (list): 
        A list containing the legends for the data series being plotted.
    lLabel (list): 
        A list containing the labels for the x-axis, y-axis, and title.

    Returns:
    None
    """
    # Create a new figure for the plot
    plt.figure()

    # Plot the data series specified by 'lHeaders' with years on the x-axis
    for head in lHeaders[1:]:   # Exclude the x-axis from the list
        plt.plot(dataframe[lHeaders[0]], dataframe[head])

    # Add labels and title to the plot
    plt.xlabel(lLabels[0])  # Label the x-axis
    plt.ylabel(lLabels[1])  # Label the y-axis
    plt.title(lLabels[2])  # Set the plot title

    # Rotate y-axis tick labels for better visibility
    plt.yticks(rotation=45)

    # Add a legend to identify the data series
    plt.legend(lLegends)

    # Remove white space on the left and right, by setting the x-axis limits
    plt.xlim(min(dataframe[lHeaders[0]]), dataframe[lHeaders[0]].max())

    # Display the plot
    plt.show()

    # Save the plot as an image file
    plt.savefig("UK_line_plot.png")

    return


def pie_chart(dataframe, lLabels, sTitle):
    """
    Create a pie chart to visualize data using custom autopct and labels.

    Args:
    dataframe (pandas.DataFrame): 
        The dataframe or data series containing the relevant data.
    lLabels (list): 
        A list of labels to be used for the pie chart segments.
    sTitle (str): 
        A string of title to be used for the pie chart.

    Returns:
    None
    """
    plt.figure()

    # Define a function to format values on the chart
    def make_autopct(dataframe):
        def my_autopct(pct):
            total = sum(dataframe)
            val = int(round(pct * total / 100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct, v=val)
        return my_autopct

    # Create the pie chart with custom autopct and font size for labels
    plt.pie(dataframe, labels=lLabels, autopct=make_autopct(
        values), textprops={'fontsize': 7})

    # Set the title for the pie chart
    plt.title(sTitle)

    # Display the pie chart
    plt.show()
    
    # Save the pie chart as an image file
    plt.savefig("UK_pie_chart.png")

    return


def bar_plot(dataframe, lHeaders, lLegends, lLabels):
    """
    Create a bar plot with dashed line to visualize two data series over time.

    Args:
    dataframe (pandas.DataFrame): 
        The DataFrame containing the data.
    lHeaders (list): 
        A list containing the x-axis and headers of the columns to plot.
    lLegends (list): 
        A list containing the legends for the data series being plotted.
    lLabel (list): 
        A list containing the labels for the x-axis, y-axis, and title.

    Returns:
    None
    """
    # Create a new figure for the plot
    plt.figure()

    # Plot the first dataseries (Population change %) with years on the x-axis
    plt.bar(dataframe[lHeaders[0]], dataframe[lHeaders[1]])

    # Plot the second data series (Inflation) with a dashed red line
    plt.plot(dataframe[lHeaders[0]], dataframe[lHeaders[2]],
             color='red', linestyle='dashed')

    # Add labels and title to the plot
    plt.xlabel(lLabels[0])  # Label the x-axis
    plt.ylabel(lLabels[1])  # Label the y-axis
    plt.title(lLabels[2])  # Set the plot title

    # Add a legend to identify the data series
    plt.legend(lLegends)

    # Display the plot
    plt.show()
    
    # Save the bar plot as an image file
    plt.savefig("UK_bar_plot.png")

    return


#  Here, we define datatypes, global or constant variables, and functions.

#  Define the URL to read data from GitHub
#  To read a file from GitHub, follow these steps:
#  1. Remove "blob" from the URL.
#  2. Replace github.com by raw.githubusercontent.com.

sUrl = 'https://raw.githubusercontent.com/shubhamvm/MScDataScience/main/'
sUrl += 'AppliedDataScience1/Assignment/Assignment%201/data/ukgs_1995_2025.csv'
# Read data from the provided URL and store it in the DataFrame df_UkSpend
df_UkSpend = pd.read_csv(sUrl)

# Display the first five rows of the DataFrame to inspect the data
print(df_UkSpend.head())

# Define the headers, legends, and labels for the first line plot
lHeader1 = ['Year', 'Education-total £ billion nominal',
            'Health Care-total £ billion nominal',
            'Pensions-total £ billion nominal',
            'Defence-total £ billion nominal']
lLegend1 = ['Education', 'Health Care', 'Pensions', 'Defence']
lLabel1 = ['Years', '£ billion nominal', 'Total spending over the years']

# Calculate the total spending for different categories
TotalEducation = df_UkSpend['Education-total £ billion nominal'].sum()
TotalHealthCare = df_UkSpend['Health Care-total £ billion nominal'].sum()
TotalPensions = df_UkSpend['Pensions-total £ billion nominal'].sum()
TotalDefence = df_UkSpend['Defence-total £ billion nominal'].sum()

# Create a list of values and labels for the pie chart
values = [TotalEducation, TotalHealthCare, TotalPensions, TotalDefence]
sTitle = 'Total spendings in £ billion'
lLabel2 = ['Education', 'Health Care', 'Pensions', 'Defence']

# Calculate the % change of UK population and round it to 2 decimal places
df_UkPop = df_UkSpend["Population-UK million"]
df_UkSpend['Population-change-%'] = df_UkPop.pct_change().mul(100).round(2)

# Define the headers, legends, and labels for the second bar plot
lHeader3 = ['Year', 'Population-change-%', 'Inflation']
lLegend3 = ['Inflation', 'Population change %']
lLabel3 = ['Years', 'percentage (%)',
           'UK Inflation and population change over the years']

# Call a function to create a line plot with the provided data and labels
line_plot(df_UkSpend, lHeader1, lLegend1, lLabel1)

# Call a function to create a pie chart with the provided data and labels
pie_chart(values, lLabel2, sTitle)

# Call a function to create a bar plot with the provided data and labels
bar_plot(df_UkSpend, lHeader3, lLegend3, lLabel3)
