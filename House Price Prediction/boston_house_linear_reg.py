#!/usr/bin/env python
# coding: utf-8

# ### importing libraries
# 
# 

# In[24]:


import pandas as pd
import numpy as np
import sklearn


# ### Loading dataset
# 

# In[25]:


path="C:\\Users\\catiq\\OneDrive\\Documents\\Downloads\\archive\\boston.csv"
df=pd.read_csv(path)


# In[26]:


df.head()


# ### Exploring dataset

# In[27]:


df.head(1)


# In[28]:


df.describe()


# #### checkng null values

# In[29]:


df.isna().sum()   #don't have any null values


# ### EDA(Exploratory data analysis)

# In[30]:


import matplotlib.pyplot as plt


# #### Strong positive corelation

# In[31]:


plt.scatter(df['RM'], df['MEDV'])  # Rooms and price
plt.xlabel("Average Number of Rooms (RM)")
plt.ylabel("House Price")
plt.title("Scatterplot: RM vs PRICE")
plt.show()


# #### not a good relaiton

# In[32]:


plt.scatter(df['CRIM'], df['MEDV'],c="g")  #crime rate and price 
plt.xlabel("Average Crimes (C)")
plt.ylabel("House Price")
plt.title("Scatterplot: Crime vs PRICE")
plt.show()


# #### strong negative correlation

# In[33]:


plt.scatter(df['LSTAT'], df['MEDV'],c="g")  #crime rate and price 
plt.xlabel("Average lower status population (C)")
plt.ylabel("House Price")
plt.title("Scatterplot: LSTAT vs PRICE")
plt.show()


# In[34]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[35]:


sns.pairplot(df[['LSTAT','MEDV','RM','CRIM']],height=2)
plt.suptitle('pairplot of features against house price prediction',y=1.02)
plt.show()


# In[36]:


# Calculate correlation matrix
corr = df.corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(corr,  annot=True, cmap='coolwarm', 
                      center=0, square=True, linewidths=0.5,
                      )

# Customize the plot
plt.title('Correlation Heatmap of boston house Dataset Features\n', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()


# Show just the correlations with PRICE (house price)
print("\nCorrelation with House Price (PRICE):")
print("=====================================")
print(corr['MEDV'].abs().sort_values(ascending=False))





# ### Train Test Split 

# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X=df.drop(columns='MEDV',axis=1)
y=df['MEDV']


# In[39]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[40]:


print(f" shape of X_train is {X_train.shape}\n shape of y_train is {y_train.shape}\n shape of X_test is {X_test.shape}\n shape of y_test is {y_test.shape}")


# ### Model training

# In[41]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# ### Make predictions

# In[42]:


predictions=regressor.predict(X_test)
print(f"predictions are:{predictions[0:10]}")


# In[43]:


y_test[0:10]


# ### Model prediction plot

# In[50]:


# Plot real vs predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=predictions, color="green", alpha=0.6, s=60)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=3)  # ideal fit line
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Real vs Predicted (Boston Housing)")
plt.show()


# ### Evaluate model

# In[44]:


from sklearn.metrics import r2_score,mean_squared_error
MSE=mean_squared_error(y_test,predictions)


# In[45]:


print(f"Mean square error is {MSE}")


# ### Model accuracy

# In[53]:


Accuracy=regressor.score(X_test,y_test)
print(f"model's accuracy is: {Accuracy}")


# In[ ]:




