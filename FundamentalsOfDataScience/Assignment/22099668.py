# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 22:34:42 2023

@author: Shubham Verma
Student ID: 22099668
email: sv23abk@herts.ac.uk
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statistics as st

# Load the dataset
url = 'https://raw.githubusercontent.com/shubhamvm/MScDataScience/main/'
url += 'FundamentalsOfDataScience/Assignment/data/data8-1.csv'
df = pd.read_csv(url, header=None, names=['Salary'], index_col=False)

# Extract data values
salary = df['Salary']

# Plot the PDF using seaborn
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot the PDF
sns.histplot(salary, kde=True, stat="density", bins=30, color="skyblue")

# Create a range of values for the x-axis
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
x = np.linspace(xmin, xmax, 100)

# Fit the data to a normal distribution
mean_value, std_dev = norm.fit(salary)
std_dev = st.stdev(salary)

# Plot the PDF using the normal distribution
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
                 alpha=0.3, label='Mean ± Std Dev Area')

plt.text(xmax * 0.6, ymax * 0.5,
         f'Mean Salary (W~): {mean_value:.2f}', fontsize=10, color='red')
plt.text(xmax * 0.6, ymax * 0.45,
         f'Standard Deviation (X): {std_dev:.2f}', fontsize=10, color='green')

# Set plot labels and title
plt.xlabel('Salary £', fontweight='bold')
plt.ylabel('Probability Density', fontweight='bold')
plt.title('Probability Density Function with Mean and Standard Deviation',
          fontweight='bold')

# Add legend
plt.legend()

# Save the plot as an image file
plt.savefig("PDF_22099668.png", dpi=300)

# Show the plot
plt.show()
