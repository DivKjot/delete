#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_1 = pd.read_csv('C:/Users/lenovo/Desktop/gani/ResearchProject/KTMK.csv')
#C:/Users/lenovo/Desktop/gani/ResearchProject/KTMK.xlsx


# In[3]:


df_1.head()


# In[4]:


print(df_1.columns)


# In[5]:


df_1.dtypes


# In[6]:


df_2 = df_1[['Date','Price']]
df_2.head()


# In[7]:


from datetime import datetime

# Convert the date column to the desired format
df_2['Date'] = pd.to_datetime(df_2['Date'], format='%b-%y').dt.strftime('%Y-%m')

df_2['Date'] = pd.to_datetime(df_2['Date'])


# In[8]:


# Remove commas and then convert 'Column_Name' to a numeric data type
df_2['Price'] = pd.to_numeric(df_2['Price'].str.replace(',', ''), errors='coerce')


# In[ ]:





# In[9]:


df_2.dtypes


# In[10]:


df_2.set_index('Date',inplace=True)


# In[48]:


df_2.head()


# In[12]:


print('shape of data',df_2.shape)


# In[13]:


df_2.plot()


# In[14]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller


# In[15]:


test_result=adfuller(df_2['Price'])


# In[16]:


# Define your function with a valid parameter name (e.g., 'close_price')

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(Price):
    result = adfuller(Price)
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis (Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root indicating it is non-stationary")

# Call the function with your 'Close Price' column as the argument
adfuller_test(df_2['Price'])


# In[17]:


#df_2 is a extracted data on which we want to perform the analysis
#for experimenting purpose we use df_3
df_3 = df_2.copy()


# In[18]:


from pmdarima.arima.utils import ndiffs


# In[19]:


ndiffs(df_3.Price,test="adf")


# In[23]:


#df_3 = pd.DataFrame(data, columns=['fi'])

# First-order differencing
df_3['first_diff'] = df_3['Price'].diff().dropna()
#df_diff = df_1.diff().dropna()


# In[36]:


df_3.head()


# In[26]:


#dropping NAN values
df_3.dropna(subset=['first_diff'], inplace=True)


# In[27]:


adfuller_test(df_3['first_diff'])


# In[ ]:


#dropping column
df_3.drop('first_diff', axis=1, inplace=True) 


# In[45]:


df_3['first_diff'].plot()


# In[28]:


from pmdarima import auto_arima


# In[33]:


#try diff order and assign a score ,Goal to minimise AIC (p,d,q)
stepwise_fit = auto_arima(df_3['first_diff'],trace =True )
stepwise_fit.summary()


# In[37]:


from statsmodels.tsa.arima_model import ARIMA


# In[46]:


from statsmodels.graphics.tsaplots import plot_acf ,plot_pacf
acf_diff = plot_acf(df_3['first_diff'])

pacf_diff = plot_pacf(df_3['first_diff'])


# In[ ]:





# In[53]:


from statsmodels.tsa.arima.model import ARIMA

# Instantiate an ARIMA model
model = ARIMA(df_3['first_diff'], order=(1,1,1))

# Fit the model
results = model.fit()

# Print the summary
print(results.summary())


# In[49]:


df_4=df_3.copy()


# In[56]:


df_4.head()


# In[54]:


print(df_4.shape)


# In[61]:


df_4['Forecast']=results.predict(start=119,end=159,dynamic=True)
df_4[['first_diff','Forecast']].plot(figsize=(9,6))


# In[72]:


print(df_3.shape)

train=df_3.iloc[:-10] #train all values except last 30

test=df_3.iloc[-10:]

print(train.shape , test.shape)


# In[73]:


train.tail()


# In[74]:


test.head()


# In[103]:


model_tr=ARIMA(train['first_diff'],order=(0,0,3))
result_tr =model_tr.fit()
result_tr


# In[109]:


start = len(train) #109
end= len(train)+len(test)-1 #109+10-1
pred = result_tr.predict(start=start,end=end)

pred.index=df_4.index[start:end+1]
print(pred)


# In[110]:


pred.plot(legend= True)
test['first_diff'].plot(legend=True)


# In[98]:


test['first_diff'].mean()


# In[101]:


result_tr()


# In[ ]:




