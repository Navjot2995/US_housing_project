#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# The data compiled for this assignment comprise two primary datasets: one for Supply and another for Demand. These datasets encompass quarterly information pertaining to essential factors affecting national home prices in the United States.

# - Supply File:
# - DATE: The date of the observation. (2003 - 2023)
# - PERMIT: New Privately-Owned Housing Units Authorized in Permit-Issuing Places: Total Units (Thousands of Units, Seasonally Adjusted Annual Rate). This variable represents the number of new housing units authorized for construction in permit-issuing places.
# - MSACSR: Monthly Supply of New Houses in the United States (Seasonally Adjusted). It indicates the monthly supply of new houses available in the United States.
# - TLRESCONS: Total Construction Spending: Residential in the United States (Millions of Dollars, Seasonally Adjusted Annual Rate). This variable represents the total construction spending on residential projects.
# - EVACANTUSQ176N: Housing Inventory Estimate: Vacant Housing Units in the United States (Thousands of Units, Not Seasonally Adjusted). It provides an estimate of the number of vacant housing units in the United States.
# - CSUSHPISA: S&P/Case-Shiller U.S. National Home Price Index (Index Jan 2000=100, Seasonally Adjusted). This variable serves as a proxy for home prices and represents the home price index for the United States.

# Demand File:
# - INTDSRUSM193N: Interest Rates, Discount Rate for United States (Billions of Dollars, Seasonally Adjusted Annual Rate). This variable represents the interest rates or discount rates for the United States.
# - UMCSENT: University of Michigan: Consumer Sentiment. It measures the consumer sentiment index based on surveys conducted by the University of Michigan.
# - GDP: Gross Domestic Product (Billions of Dollars, Seasonally Adjusted Annual Rate).
# - MORTGAGE15US : 30-Year Fixed Rate Mortgage Average in the United States (Percent, Not Seasonally Adjusted). It indicates the average interest rate for a 30-year fixed-rate mortgage.
# - MSPUS: Median Sales Price of Houses Sold for the United States (Not Seasonally Adjusted)
# - CSUSHPISA: S&P/Case-Shiller U.S. National Home Price Index (Index Jan 2000=100, Seasonally Adjusted). This variable serves as a proxy for home prices and represents the home price index for the United States.

# The data in both the Supply and Demand files is observed quarterly, meaning that data points are available at the conclusion of each quarter. The S&P Case-Shiller U.S. National Home Price Index (CSUSHPISA) is employed as a surrogate for home prices and serves as the dependent variable for the analysis. 
#     This index is reported on a quarterly basis and undergoes seasonal adjustment, providing a comprehensive gauge of nationwide home prices. These datasets offer valuable insights into various supply-demand factors and their potential impact on home prices in the United States. 
#     Analyzing this data and constructing a data science model will facilitate the exploration of relationships between these factors and fluctuations in home prices over the last two decades.

# In[76]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


# In[86]:


import pandas as pd

# Load supply and demand datasets (replace with your file paths)
supply_data = pd.read_csv('supply.csv')
demand_data = pd.read_csv('demand.csv')

# Merge datasets based on the 'DATE' column
merged_data = pd.merge(demand_data, supply_data, on='DATE', how='inner')

# Display the merged dataset
print(merged_data.head())

# Save the merged dataset to a new CSV file (optional)
merged_data.to_csv('merged_data.csv', index=False)


# In[87]:


supply_data['DATE'] = pd.to_datetime(supply_data['DATE'])
demand_data['DATE'] = pd.to_datetime(demand_data['DATE'])

supply_data = supply_data.sort_values('DATE')
demand_data = demand_data.sort_values('DATE')

merged_data = pd.merge(supply_data, demand_data, on='DATE', suffixes=('_supply', '_demand'))

merged_data.dropna(subset=['MSACSR', 'PERMIT', 'TLRESCONS', 'EVACANTUSQ176N', 'MORTGAGE30US', 'GDP', 'UMCSENT'], inplace=True)

imputer = SimpleImputer(strategy='mean')
merged_data['INTDSRUSM193N'] = imputer.fit_transform(merged_data[['INTDSRUSM193N']])

merged_data = merged_data.reset_index(drop=True)


# In[88]:


merged_data.head()


# In[89]:


merged_data.drop('CSUSHPISA_supply', axis=1, inplace=True)

merged_data.rename(columns={'CSUSHPISA_demand': 'CSUSHPISA'}, inplace=True)
merged_data['CSUSHPISA'] = merged_data['CSUSHPISA'].fillna(merged_data['CSUSHPISA'].mean())


# #### Data Desciption

# In[90]:


merged_data.info()


# In[91]:


merged_data.describe()


# In[ ]:


merged_data.dropna(inplace=True)
merged_data.head()


# In[93]:


correlation = merged_data.corr()['CSUSHPISA']
correlation_table = pd.DataFrame(correlation).reset_index()
correlation_table.columns = ['Factors', 'Correlation with CSUSHPISA']
print(correlation_table)


# In[94]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = merged_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')


# In[9]:


# Time series plot
plt.figure(figsize=(12, 6))
plt.plot(merged_data['DATE'], merged_data['CSUSHPISA_x'], label='CSUSHPISA', marker='o')
plt.xlabel('Date')
plt.ylabel('CSUSHPISA')
plt.title('Time Series Plot of CSUSHPISA')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[65]:


merged_data.columns


# In[98]:


# Data distribution plots
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(data=merged_data, x='CSUSHPISA', bins=20, kde=True)
plt.title('Distribution of CSUSHPISA')

plt.subplot(2, 2, 2)
sns.histplot(data=merged_data, x='MSACSR', bins=20, kde=True)
plt.title('Distribution of MSACSR')

plt.subplot(2, 2, 3)
sns.histplot(data=merged_data, x='GDP', bins=20, kde=True)
plt.title('Distribution of GDP')

plt.subplot(2, 2, 4)
sns.histplot(data=merged_data, x='MORTGAGE30US', bins=20, kde=True)
plt.title('Distribution of MORTGAGE30US')

plt.tight_layout()
plt.show()


# In[12]:


# Pairplot
sns.pairplot(merged_data, vars=['CSUSHPISA_x', 'MSACSR', 'GDP', 'MORTGAGE30US'])
plt.suptitle('Pairplot of Key Variables', y=1.02)


# In[96]:


correlation = merged_data.corr()['CSUSHPISA']
correlation_table = pd.DataFrame(correlation).reset_index()
correlation_table.columns = ['Factors', 'Correlation with CSUSHPISA']
print(correlation_table)


# MORTGAGE30US: There is a moderately negative correlation of approximately -0.218493 between 'MORTGAGE30US' (30-Year Fixed Mortgage Rate) and 'CSUSHPISA'. This suggests that, in general, as mortgage rates increase, the housing price index tends to decrease, which is a common trend in real estate.
# 
# UMCSENT: 'UMCSENT' (University of Michigan Consumer Sentiment Index) has a weak negative correlation of approximately -0.097450 with 'CSUSHPISA'. This implies that consumer sentiment is not strongly associated with changes in the housing price index.
# 
# INTDSRUSM193N: 'INTDSRUSM193N' (Effective Federal Funds Rate) has a positive correlation of approximately 0.157768 with 'CSUSHPISA'. This suggests that as the federal funds rate increases, housing prices tend to rise as well, although the correlation is relatively weak.
# 
# MSPUS: 'MSPUS' (Median Sales Price of Houses Sold) has a very strong positive correlation of approximately 0.939512 with 'CSUSHPISA'. This indicates a close relationship between the median sales price of houses and the housing price index. When house prices go up, the housing price index tends to increase as well.
# 
# GDP: 'GDP' (Gross Domestic Product) exhibits a strong positive correlation of approximately 0.855098 with 'CSUSHPISA'. This suggests that there is a strong connection between the overall economic performance, as measured by GDP, and housing prices. When the economy grows, housing prices tend to rise.
# 
# MSACSR: 'MSACSR' (Mortgage Delinquency Rate) has a relatively weak positive correlation of approximately 0.121782 with 'CSUSHPISA'. While there is a positive relationship, it's not very strong, implying that changes in mortgage delinquency rates have a limited impact on the housing price index.
# 
# PERMIT: 'PERMIT' (Housing Units Authorized by Building Permits) has a moderate positive correlation of approximately 0.382354 with 'CSUSHPISA'. This suggests that an increase in housing permits is associated with higher housing prices, but the relationship is not extremely strong.
# 
# TLRESCONS: 'TLRESCONS' (Total Residential Construction Spending) has a very strong positive correlation of approximately 0.882204 with 'CSUSHPISA'. This indicates that when there is more spending on residential construction, housing prices tend to rise, reflecting the demand for new housing.
# 
# EVACANTUSQ176N: 'EVACANTUSQ176N' (Quarterly Homeownership Vacancy Rate) has a strong negative correlation of approximately -0.592952 with 'CSUSHPISA'. This implies that when the vacancy rate in homeownership increases, housing prices tend to decrease. High vacancy rates can indicate a weaker housing market.

# In[66]:


merged_data['DATE'] = pd.to_datetime(merged_data['DATE'])
merged_data.set_index('DATE', inplace=True)

merged_data['MSACSR'] = pd.to_numeric(merged_data['MSACSR'], errors='coerce')
merged_data['PERMIT'] = pd.to_numeric(merged_data['PERMIT'], errors='coerce')
merged_data['TLRESCONS'] = pd.to_numeric(merged_data['TLRESCONS'], errors='coerce')
merged_data['EVACANTUSQ176N'] = pd.to_numeric(merged_data['EVACANTUSQ176N'], errors='coerce')


# #### Bivariate Analysis of the Datset

# In[101]:



# Define the list of variables for bivariate analysis
variables_of_interest = ["MORTGAGE30US", "UMCSENT", "INTDSRUSM193N", "MSPUS", "GDP"]

# Loop through each variable and create scatter plots
for variable in variables_of_interest:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=merged_data[variable], y=merged_data["CSUSHPISA"], alpha=0.7)
    plt.title(f"Scatter Plot: {variable} vs CSUSHPISA")
    plt.xlabel(variable)
    plt.ylabel("CSUSHPISA")
    plt.grid(True)
    plt.show()


# In[ ]:





# Scaling and Model BUilding Process Starts here

# In[102]:


# Initialize the Min-Max scaler
scaler = MinMaxScaler()

# Perform Min-Max scaling on the entire DataFrame (excluding 'CSUSHPISA')
data_scaled = merged_data.copy()
data_scaled[merged_data.columns[1:]] = scaler.fit_transform(merged_data[merged_data.columns[1:]])


# In[103]:


features = ['MSACSR', 'PERMIT', 'TLRESCONS', 'EVACANTUSQ176N', 'MORTGAGE30US', 'GDP', 'UMCSENT', 'INTDSRUSM193N', 'MSPUS']
target = 'CSUSHPISA'

X_train, X_test, y_train, y_test = train_test_split(data_scaled[features], data_scaled[target], test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Neural Network': MLPRegressor()
}

results = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores = -scores
    avg_mse = mse_scores.mean()
    results[model_name] = avg_mse


best_model = min(results, key=results.get)
best_model_instance = models[best_model]


best_model_instance.fit(X_train, y_train)


predictions = best_model_instance.predict(X_test)
mse = mean_squared_error(y_test, predictions)


print("Model Selection Results:")
for model, mse_score in results.items():
    print(f"{model}: MSE={mse_score}")

print(f"\nBest Model: {best_model}")
print(f"Best Model MSE on Testing Set: {mse}")


# In[104]:


from sklearn.metrics import r2_score
r2 = r2_score(y_test, predictions)

print("R-squared score:", r2)


# In[105]:


best_model_instance.fit(X_train, y_train)

coefficients = best_model_instance.coef_


print("Coefficients:")
for feature, coefficient in zip(features, coefficients):
    print(f"{feature}: {coefficient}")


# #### Model Evaluation:

# Here's an insight based on the coefficients of your regression model:
# 
# MSACSR (Monthly Supply of Houses): A positive coefficient of 0.376 suggests that an increase in the monthly supply of houses is associated with an increase in the dependent variable (CSUSHPISA), which represents U.S. home prices. This indicates that when the supply of houses increases, it can lead to higher home prices.
# 
# PERMIT (Housing Permits): With a positive coefficient of 0.192, it implies that a higher number of housing permits issued is linked to higher U.S. home prices. This suggests that an increase in housing permits may be indicative of a growing demand for housing.
# 
# TLRESCONS (Total Residential Construction Spending): This feature has a positive coefficient of 0.229, indicating that increased spending on total residential construction is associated with higher U.S. home prices. This makes sense as higher construction spending can lead to more housing availability and potentially drive up prices.
# 
# EVACANTUSQ176N (Quarterly Vacancy Rate): The negative coefficient of -0.040 suggests that a higher vacancy rate is associated with lower U.S. home prices. This implies that an oversupply of vacant housing units can put downward pressure on home prices.
# 
# MORTGAGE30US (30-Year Mortgage Rate): A negative coefficient of -0.336 indicates that higher 30-year mortgage rates are linked to lower U.S. home prices. When mortgage rates rise, it can make buying a home less affordable, potentially reducing demand and prices.
# 
# GDP (Gross Domestic Product): The negative coefficient of -0.267 suggests that an increase in GDP is associated with lower U.S. home prices. This relationship might be due to economic factors affecting housing demand and supply.
# 
# UMCSENT (University of Michigan Consumer Sentiment Index): With a negative coefficient of -0.046, it implies that lower consumer sentiment is linked to lower U.S. home prices. When consumers are less optimistic about the economy, they may be less likely to make large purchases like homes.
# 
# INTDSRUSM193N (Effective Federal Funds Rate): The positive coefficient of 0.137 suggests that higher federal interest rates are associated with higher U.S. home prices. This is somewhat counterintuitive, as higher interest rates typically mean higher borrowing costs, which can reduce demand for homes.
# 
# MSPUS (Median Sales Price of Houses Sold): This feature has the highest positive coefficient of 0.769, indicating a strong positive correlation with U.S. home prices. An increase in the median sales price of houses sold is strongly associated with higher U.S. home prices.
# 
# In summary, this model suggests that factors related to housing supply (MSACSR, PERMIT, and TLRESCONS) tend to drive up U.S. home prices. On the other hand, factors like mortgage rates (MORTGAGE30US) and GDP have a negative impact on home prices. The relationship between interest rates (INTDSRUSM193N) and home prices appears counterintuitive and may require further investigation. Additionally, the median sales price of houses sold (MSPUS) is a strong predictor of U.S. home prices.

# ##### Conclusion:
# ##### Based on the correlation analysis and the coefficients from the Linear Regression model, several key insights can be derived:
# 
# - Supply factors, such as house inventory and the number of authorized housing units, have a positive influence on home prices. Higher construction spending on residential projects also contributes significantly to higher home prices.
# - Demand factor, such as mortgage interest rates, have a negative impact on home prices. Higher mortgage rates and lower consumer sentiment are associated with slightly lower home prices.
# - Economic factors, including GDP and interest rates, play a crucial role in determining home prices. A strong economy with higher GDP and slightly lower interest rates tends to support higher home prices.
# - The median sales price of houses sold is strongly correlated with home prices, reflecting the importance of market dynamics and buyer behaviour in determining home price movements.
# - These insights can be valuable for various stakeholders in the real estate market, including home buyers, sellers, developers, and policymakers. Understanding the factors that influence home prices can help make informed decisions related to investments, financing, and economic policies.

# In[ ]:




