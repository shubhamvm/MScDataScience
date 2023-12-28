# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 10:37:30 2023

@author: Shubham Verma
Student ID: 22099668
email: sv23abk@herts.ac.uk
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def pdf_distribution_plot(df1, labels):
    # Plot 1: Histogram - Total bill distribution
    """
    Create a pdf distribution plot with mean and starndard deviation to visualize data.

    Args:
    df1 (pandas.DataFrame): The DataFrame containing the data.
    labels (list): A list containing the labels for the x-axis, y-axis, and title.

    Plots the historgram of data with pdf, mean and starndard deviation

    Returns:
    None
    """
    
    plt.figure(figsize=(12, 6))

    # Plot the PDF
    sns.histplot(df1, kde=True, stat="density", bins=30, color="skyblue")

    # Create a range of values for the x-axis
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    x = np.linspace(xmin, xmax, 100)

    # Fit the data to a normal distribution
    mean_value, std_dev = norm.fit(df1)

    # Plot 1 the PDF using the normal distribution
    pdf = norm.pdf(x, mean_value, std_dev)
    plt.plot(x, pdf, 'k', linewidth=2)

    # Plot mean and standard deviation lines
    plt.axvline(mean_value, color='red', linestyle='dashed',
                linewidth=2, label=f'Mean = {mean_value:.2f}')
    plt.axvline(mean_value - std_dev, color='green', linestyle='dashed',
                linewidth=2, label=f'Mean - Std Dev = {(mean_value - std_dev):.2f}')
    plt.axvline(mean_value + std_dev, color='green', linestyle='dashed',
                linewidth=2, label=f'Mean + Std Dev = {(mean_value + std_dev):.2f}')

    # Fill the area between mean - std_dev and mean + std_dev with color
    x_fill = np.linspace(mean_value - std_dev, mean_value + std_dev, 100)
    y_fill = np.exp(-(x_fill - mean_value)**2 / (2 * std_dev**2)) / \
        (std_dev * np.sqrt(2 * np.pi))
    plt.fill_between(x_fill, y_fill, color='red',
                     alpha=0.4, label='Mean ± Std Dev Area')

    plt.text(xmax * 0.6, ymax * 0.5,
             f'Mean: {mean_value:.2f}', fontsize=10, color='red')
    plt.text(xmax * 0.6, ymax * 0.45,
             f'Standard Deviation: {std_dev:.2f}', fontsize=10, color='green')

    # Set plot labels and title
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[2], fontweight='bold')

    # Add legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig(student_id + "P1_PDF.png", dpi=300)

    # Show the plot
    plt.show()

    return None


def bar_plot(df1, headers, labels):
    # Plot 2 bar plot
    """
    Create a bar plot with mean and starndard deviation to visualize data.
    
    Args:
    df1 (pandas.DataFrame): The DataFrame containing the data.
    headers (list): A list containing the headers for plotting.
    labels (list): A list containing the labels for the x-axis, y-axis, and title.
    
    Plots the historgram of data with pdf, mean and starndard deviation
    
    Returns:
    None
    """
    # Plot gender-wise total bills over the days in a bar chart
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=headers[0], y=headers[1], hue=headers[2],
                     data=df1, errorbar=None, palette='Set1')

    # Add labels and title
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(labels[2], fontweight='bold')

    # Add total bill amount text on each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='center', xytext=(0, -100), textcoords='offset points', fontsize=10)

    # Add a legend
    plt.legend()

    # Save the plot as an image file
    plt.savefig(student_id + "P2_Bar.png", dpi=300)

    # Show the plot
    plt.show()

    return None


def pie_plot(df1, headers, labels):
    """
    Create a pie chart and nested pie chart to visualize data.
    
    Args:
    df1 (pandas.DataFrame): The DataFrame containing the data.
    headers (list): A list containing the headers for plotting.
    labels (list): A list containing the labels for the title.
    
    Plots the pie chart and nested pie chart to visualize data.
    
    Returns:
    None
    """
    
    # Create a figure with subplots (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    gender_distribution = df1[headers[0]].value_counts()
    explode = (0.1, 0)
    axes[0].pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%',
                startangle=90, explode=explode, colors=['lightcoral', 'lightskyblue'])
    axes[0].set_title(labels[0], fontweight='bold')
    axes[0].legend()

    # Plot 3,2: Nested Pie chart for gender and time distribution
    gender_time_distribution = df1.groupby(headers).size().unstack()

    # Outer Circle (Gender Distribution)
    size_outer = gender_time_distribution.sum(axis=1)
    outside_pie, outside_text, _ = axes[1].pie(size_outer, labels=gender_time_distribution.index, autopct='%1.1f%%', startangle=90, colors=[
                                               '#66b3ff', '#ffcc99'], wedgeprops=dict(width=0.3), labeldistance=1.05, pctdistance=0.9, textprops=dict(color="black"))

    # Inner Circle (Time Distribution within Gender)
    size_inner = gender_time_distribution.values.flatten()
    labels_inner = [
        f"{gender} - {time}" for gender in gender_time_distribution.index for time in gender_time_distribution.columns]
    inside_pie, inside_text, _ = axes[1].pie(size_inner, autopct='%1.1f%%', startangle=90, colors=['#99ff99', '#ff6666'], radius=0.6, wedgeprops=dict(
        width=0.3), labels=labels_inner, labeldistance=0.75, pctdistance=0.6, textprops=dict(color="black", fontsize=8))

    # Set aspect ratio to be equal to ensure a circular shape
    axes[1].set_aspect('equal')

    # Set title for subplot 2
    axes[1].set_title(labels[1], fontweight='bold')
    axes[1].legend()

    # Save the plot as an image file
    plt.savefig(student_id + "P3_Pie.png", dpi=300)

    plt.show()

    return None


def violin_plot(df1, headers, labels):
    """
    Create a violin chart to visualize data as per headers.
    
    Args:
    df1 (pandas.DataFrame): The DataFrame containing the data.
    headers (list): A list containing the headers for plotting.
    labels (list): A list containing the labels for the x-axis, y-axis, and title.
    
    Plots the violin chart to visualize data as per headers.
    
    Returns:
    None
    """
    plt.figure(figsize=(12, 6))

    sns.violinplot(x=headers[0], y=headers[1],
                   hue=headers[2], data=df1, split=True)
    plt.title(labels[0], fontweight='bold')

    # Save the plot as an image file
    plt.savefig(student_id + "P4_Violin.png", dpi=300)

    plt.show()

    return None


# Load the example 'tips' dataset from seaborn
df = sns.load_dataset('tips')

#student id for saving the plots
student_id ='22099668_'

# Summary Statistics
summary_stats = df.describe()
# Calculate the mean total bill for each day
mean_total_bill = df.groupby('day')['total_bill'].mean()
all_bills_mean = np.mean(df['total_bill'])

df_bills_pdf = df['total_bill'].dropna()
labels_pdf = ['Total bill', 'Probability Density',
              'Probability Density Function with Mean and Standard Deviation']
pdf_distribution_plot(df_bills_pdf, labels_pdf)

headers_bar = ['day', 'total_bill', 'sex']
labels_bar = ['Day', 'Total Bill Amount', 'Gender-wise Total Bills over Days']
bar_plot(df, headers_bar, labels_bar)

headers_pie = ['sex', 'time']
labels_pie = ['Gender Distribution', 'Two-Level Nested Pie Chart for Gender and Time Distribution']
pie_plot(df, headers_pie, labels_pie)

headers_violin = ['day', 'total_bill', 'sex']
labels_violin = ['Violin plot - Total bill distribution by time and day']
violin_plot(df, headers_violin, labels_violin)
