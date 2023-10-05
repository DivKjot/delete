#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


df = pd.read_csv('C:/Users/lenovo/Downloads/25-09-2022-TO-25-09-2023-RELIANCE-ALL-N.csv')


# In[6]:


df.head()


# In[7]:


df.tail()


# In[19]:



print(df.columns)


# In[16]:


df.dtypes


# In[21]:


df_1 = df[['Date  ', 'Close Price  ']]
df_1.head()


# In[28]:


# Convert Month into Datetime

df_1['Date  '] = pd.to_datetime(df_1['Date  '])

#df_1['Date  '] = pd.to_datetime(df_1['Date  '].str.strip())


# In[36]:


# Remove commas and then convert 'Column_Name' to a numeric data type
df_1['Close Price  '] = pd.to_numeric(df_1['Close Price  '].str.replace(',', ''), errors='coerce')


# In[ ]:





# In[37]:


df_1.dtypes


# In[38]:


df_1.head()


# In[32]:


df_1.set_index('Date  ',inplace=True)


# In[33]:


df_1.head()


# In[39]:


df_1.plot()


# In[40]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller


# In[41]:


test_result=adfuller(df_1['Close Price  '])


# In[45]:


# Define your function with a valid parameter name (e.g., 'close_price')

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(close_price):
    result = adfuller(close_price)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis (Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root indicating it is non-stationary")

# Call the function with your 'Close Price' column as the argument
adfuller_test(df_1['Close Price  '])


# DIFFERENCING

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


pip install --no-cache-dir pmdarima


# In[61]:


from pmdarima.arima.utils import ndiffs


# In[64]:


#df.rename(columns={'Old_Column_Name': 'New_Column_Name'}, inplace=True)
df_1.rename(columns={'Close Price  ':'Close_Price'},inplace = True)


# In[65]:


ndiffs(df_1.Close_Price,test="adf")


# In[66]:


df_diff = df_1.diff().dropna()
df_diff.plot()


# In[67]:


adfuller_test(df_diff)


# ACF AND PACF MODELS

# In[69]:


from statsmodels.graphics.tsaplots import plot_acf ,plot_pacf


# In[70]:


acf_diff = plot_acf(df_diff)

pacf_diff = plot_pacf(df_diff)


# In[73]:


from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_1,order =(1,1,1))
model_fit = model.fit()
print(model_fit.summary())


# PLOT

# In[ ]:





# In[106]:


df_1['Forecast']=model_fit.predict(start=230,end=249,dynamic=True)
df_1[['Close_Price','Forecast']].plot(figsize=(12,8))


# In[113]:


# Make predictions using the model for the desired number of periods
forecast_test = model_fit.forecast(steps=len(df_1))  # Adjust the number of steps as needed

# Create a new column 'forecast_manual' in df_1 with the forecasted values
df_1['forecast_manual'] = [None]*len(df_1)+list(forecast_test)

# Plot the DataFrame
df_1.plot(figsize=(12, 8))


# In[107]:


forecast_test = model_fit.forecast(len(df_1))
df_1['forecast_manual']=[None]*len(df_1)+list(forecast_test)
#df_1.plot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


df_1.tail()


# In[111]:


# Assuming you want to drop the 'Forecast' column from 'df_1'
df_1.drop(columns=['forecast_manual'], inplace=True)


# In[96]:


df_1.dtypes


# In[98]:


df_1.columns


# In[ ]:




