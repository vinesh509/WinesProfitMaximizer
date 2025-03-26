*WinesProfitMaximizer*
# Predictive Pricing Strategy for Profit Maximization of Wine Sales 🍷

## Project Overview  
**Business Challenge**:  
Atomic Wines, a Midwest retail chain, sought to identify **underpriced wines** from its wholesaler’s catalog to maximize profit margins. Wines not currently in inventory were suspected to have untapped value based on their chemical properties and predicted quality.  

**Solution**:  
- Built a **predictive model** using expert-rated quality scores and chemical data from existing inventory (`Wines Sold.csv`).  
- Applied the model to the wholesaler’s `Wines Remaining.csv` dataset to predict quality for unsold wines.  
- Identified **top 20 high-value wines** using a **quality-to-price ratio** metric for strategic purchasing.  

## Key Features  
- **Data Preprocessing**: Handled outliers, missing values, and feature engineering.  
- **Predictive Modeling**: Linear regression to predict wine quality from chemical properties.  
- **Value Scoring**: Ranked wines by `Predicted Quality / Price` to highlight markup potential.  
- **Automated Outputs**: Generated CSV files for easy integration into business workflows.
  
# Installation  

**Clone the repository**:  
   ```bash  
   git clone https://github.com/vinesh509/WinesProfitMaximizer.git  
   cd WinesProfitMaximizer
   ```
**Dependencies**: 
   ```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```
# Download datasets
`Place Wines Sold.csv and Wines Remaining.csv in the project root`

**Run the code** :
```bash
python3 atomicwines.py
```
# Outputs:

- `Out_data/selected_wines.csv`: Top 20 underpriced wines recommended for purchase for profit maximization.
- `Out_data/wine_data_train.csv`: Trained dataset before outliner removal.
- `Out_data/wine_data_no_outs_train.csv`: Trained data after outlier removal.

# Methodology

**Exploratory Data Analysis (EDA)**:
- Analyzed correlations between chemical properties and quality.
- Visualized relationships, alcohol vs quality, shown in this, for instance.

**Model Development**:
- Trained a linear regression model on 75% of Wines Sold data.
- Validated performance using Mean Absolute Error (MAE) and R-squared.

**Selection Criteria**:
- Predicted quality for Wines Remaining.
- Ranked wines by Predicted Quality / Price and selected top 20.

# Results

**Top 20 Wines**:
Identified wines with high predicted quality relative to their price (e.g., a wine with quality=7 priced at $6.99).

**Model Performance**:
- MAE: 0.52 (average error in quality prediction).
- R²: 0.72 (72% of quality variance explained by chemical properties).

# Future Enhancements Roadmap 🚀

**Real-Time Price & Demand Monitoring**  
- Dynamic Pricing Engine: Integrate APIs to track live wholesaler pricing and regional sales trends, enabling instant identification of price drops or demand spikes.  
- Automated Alerts: Notify procurement teams when underpriced wines match predicted quality thresholds.  

**Customer-Centric Quality Scoring**  
- Hybrid Quality Metrics: Combine expert ratings with customer reviews (from Atomic’s sales platforms) to refine quality predictions.  
- Seasonal Preference Modeling: Adjust recommendations based on regional Midwest buying patterns (e.g., hearty reds in winter, crisp whites in summer).  

**Sustainability-Driven Selection**  
- Eco-Scoring System: Prioritize wines with certifications like organic, biodynamic, or low-carbon production.  
- Ethical Sourcing Flags: Highlight wines from fair-trade vineyards or minority-owned distributors.  

**Advanced Predictive Techniques**  
- Ensemble Modeling: Replace linear regression with XGBoost or LightGBM to capture non-linear relationships between chemical properties and quality.  
- Explainable AI (XAI): Add SHAP values to clarify why specific wines are recommended (e.g., "Selected for high alcohol + low volatile acidity").  

# License & Contact:
- *This project is licensed under the MIT License. See LICENSE for details.*
- *Developed by: Vinesh Vangapandu*
- *📧 Email: vinesh509@gmail.com*
- *🔗 LinkedIn: www.linkedin.com/in/vinesh-vangapandu-783590160*
