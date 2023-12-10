# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 17:58:52 2023

@author: Shubham Verma
Student ID: 22099668
email: sv23abk@herts.ac.uk
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load the example 'tips' dataset from seaborn
df = sns.load_dataset('tips')

# Summary Statistics
summary_stats = df.describe()

# Plot 1: Histogram - Total bill distribution
plt.figure(figsize=(12, 6))
bills = df['total_bill'].dropna()

# Plot the PDF
sns.histplot(bills, kde=True, stat="density", bins=30, color="skyblue")

# Create a range of values for the x-axis
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
x = np.linspace(xmin, xmax, 100)

# Fit the data to a normal distribution
mean_value, std_dev = norm.fit(bills)

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
plt.xlabel('Total bill')
plt.ylabel('Probability Density')
plt.title('Probability Density Function with Mean and Standard Deviation',
          fontweight='bold')

# Add legend
plt.legend()

# Save the plot as an image file
plt.savefig("22099668_P1_PDF.png", dpi=300)

# Show the plot
plt.show()

# Plot 2
# Calculate the mean total bill for each day
mean_total_bill = df.groupby('day')['total_bill'].mean()
all_bills_mean = np.mean(df['total_bill'])

# Plot gender-wise total bills over the days in a bar chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='day', y='total_bill', hue='sex', data=df, errorbar=None, palette='Set1')

# Add labels and title
plt.xlabel('Day')
plt.ylabel('Total Bill Amount')
plt.title('Gender-wise Total Bills over Days', fontweight='bold')

# Add total bill amount text on each bar
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.2f}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, -100), textcoords='offset points', fontsize=10)

# Add a legend
plt.legend()

# Save the plot as an image file
plt.savefig("22099668_P2_Bar.png", dpi=300)

# Show the plot
plt.show()

# Plot 3: Exploded Pie chart for gender distribution
# Create a figure with subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

gender_distribution = df['sex'].value_counts()
explode = (0.1, 0)
axes[0].pie(gender_distribution, labels=gender_distribution.index, autopct='%1.1f%%',
            startangle=90, explode=explode, colors=['lightcoral', 'lightskyblue'])
axes[0].set_title('Gender Distribution', fontweight='bold')
axes[0].legend()

# Plot 3,2: Nested Pie chart for gender and time distribution
gender_time_distribution = df.groupby(['sex', 'time']).size().unstack()

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
axes[1].set_title(
    'Two-Level Nested Pie Chart for Gender and Time Distribution', fontweight='bold')
axes[1].legend()



# Save the plot as an image file
plt.savefig("22099668_P3_Pie.png", dpi=300)

plt.show()

# Plot 4: Violin plot - Total bill distribution by time and day

plt.figure(figsize=(12, 6))
sns.violinplot(x='day', y='total_bill', hue='time', data=df, split=True)
plt.title('Violin plot - Total bill distribution by time and day', fontweight='bold')
# Save the plot as an image file
plt.savefig("22099668_P4_Violin.png", dpi=300)
plt.show()
