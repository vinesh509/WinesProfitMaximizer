#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atomic Wines Case Solution - Revised
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm

# Load Data
wine_data = pd.read_csv('/Users/vineshvangapandu/Desktop/Atomic_wines/In_data/Wines Sold.csv')


# Average price for quality 6
group_df = wine_data.groupby('quality').agg(
    count=('Brand_ID', 'count'), 
    avg_price=('Price', 'mean'),
).reset_index()

atom_qlt_6 = group_df.loc[group_df['quality'] == 6, 'avg_price'].values[0]
print(f"Average price for quality 6: ${atom_qlt_6:.2f}")

# Highest residual sugar
avg_sg_res = wine_data.groupby('quality')['residual_sugar'].mean()
qlt_lv = avg_sg_res.idxmax()
print(f"Quality with highest residual sugar: {qlt_lv}")

# Correlation matrix
chem_prop_vars = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                  'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                  'pH', 'sulphates', 'alcohol']  
corr_matrix = wine_data[['quality'] + chem_prop_vars].corr()
high_corr = corr_matrix['quality'].drop('quality').abs().nlargest(2)
print(f"Highest correlations: {high_corr.index.tolist()}")

# Scatterplot

top_corr_var = high_corr.index[0]
sns.scatterplot(x=top_corr_var, y='quality', data=wine_data)
plt.title(f"{top_corr_var} vs Quality")
plt.show()

# Train/Test Split

train_df, test_df = train_test_split(wine_data, test_size=0.25, random_state=42)
print(f"Data split into 75% train, 25% test")

# Regression Model

X_train = sm.add_constant(train_df[chem_prop_vars])
y_train = train_df['quality']
model_all = sm.OLS(y_train, X_train).fit()
print(f"Alcohol p-value: {model_all.pvalues['alcohol']:.4f}")

# Removing Density
X_train_no_density = X_train.drop(columns=['density'])
model_no_density = sm.OLS(y_train, X_train_no_density).fit()
print(f"Alcohol t-value change: {model_all.tvalues['alcohol']:.2f} -> {model_no_density.tvalues['alcohol']:.2f}")

# Model Improvement
print(f"Adj. R²: {model_all.rsquared_adj:.3f} vs {model_no_density.rsquared_adj:.3f}")

# Best Model
final_vars = ['alcohol', 'volatile_acidity', 'sulphates', 'residual_sugar']
model_final = LinearRegression().fit(train_df[final_vars], y_train)
y_pred = model_final.predict(train_df[final_vars])
mae = mean_absolute_error(y_train, y_pred)
print(f"MAE: {mae:.2f}, Variables: {final_vars}")

# Removing Outliers
numeric_cols = ['Price', 'quality'] + chem_prop_vars
Q1 = train_df[numeric_cols].quantile(0.25)
Q3 = train_df[numeric_cols].quantile(0.75)
IQR = Q3 - Q1
outlier_mask = ~((train_df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                (train_df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
wine_data_no_outs = train_df[outlier_mask]
print(f"Outliers removed: {len(train_df) - len(wine_data_no_outs)}")

# Clean Model

model_clean = LinearRegression().fit(wine_data_no_outs[final_vars], wine_data_no_outs['quality'])
mae_clean = mean_absolute_error(wine_data_no_outs['quality'], 
                               model_clean.predict(wine_data_no_outs[final_vars]))
print(f"Clean MAE: {mae_clean:.2f}")

# Export Data

os.makedirs('Out_data', exist_ok=True)
train_df.to_csv(os.path.join('Out_data', 'wine_data_train.csv'), index=False)
wine_data_no_outs.to_csv(os.path.join('Out_data', 'wine_data_no_outs_train.csv'), index=False)

# Wine Selection
wines_rem = pd.read_csv('/Users/vineshvangapandu/Desktop/Atomic_wines/In_data/Wines Remaining.csv')

# Predict quality
X_rem = wines_rem[final_vars]
pred_quality = np.round(model_final.predict(X_rem)).astype(int)
wines_rem['predicted_quality'] = np.where(pred_quality < 3, 3, pred_quality)

wines_rem['value_score'] = wines_rem['predicted_quality'] / wines_rem['Price']
selected_wines = wines_rem.nlargest(20, 'value_score')

output_columns = [
    'Brand_ID', 'Classification', 'Price', 'fixed_acidity', 'volatile_acidity',
    'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
    'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol',
    'free_chlorides', 'predicted_quality'
]

selected_wines = selected_wines[output_columns].sort_values('Brand_ID').reset_index(drop=True)

float_cols = selected_wines.select_dtypes(include=['float']).columns.difference(['Price'])
selected_wines[float_cols] = selected_wines[float_cols].round(4)
selected_wines['predicted_quality'] = selected_wines['predicted_quality'].astype(int)

selected_wines.to_csv(os.path.join('Out_data', 'selected_wines.csv'), index=False)

