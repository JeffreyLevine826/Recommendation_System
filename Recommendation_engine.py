# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:23:20 2024

@author: Jeffrey Levine
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline  

# Load the data

df = pd.read_csv('Data/ratings_Beauty.csv')

print("Shape: %s" % str(df.shape))
print("Column name: %s" % str(df.columns))

print(df.head())

# Unique Users and Products
print("Unique UserID count: %s" % str(df.UserId.nunique()))
print("Unique ProductID count: %s" % str(df.ProductId.nunique()))

# Rating frequency
sns.countplot(data=df, x='Rating', palette=sns.color_palette('Blues'))
plt.show()

# Data Wrangling
# Creating fields and measures from existing data
# This helps generate more data points and validates the ideology

# Mean rating for each product
product_rating = df.groupby('ProductId')['Rating'].mean()
print(product_rating.head())

# Mean reating KDE distribution
sns.kdeplot(product_rating, shade=True, color='grey')
plt.show()

""" We can notice a large spike in the mean rating at value 5. This is a valuable indicator
    that points to the skewness of the data. Hence we need to further analyse this issue.
"""
# Count of the number of ratings per Product
product_rating_count = df.groupby('ProductId')['Rating'].count()
print(product_rating_count.head())

# Number of ratings per product KDE distribution
sns.kdeplot(product_rating_count, shade=True, color='grey')
plt.show()



