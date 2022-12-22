#!/usr/bin/env python
# coding: utf-8

# ## **Linear Regression with Python Scikit Learn**
# In this section we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions. We will start with simple linear regression involving two variables.
# 
# ### **Simple Linear Regression**
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# ## Author : Saragadam Abhiram Laxmi Raj
# ## Task 1  : Prediction using Supervised Machine Learning
# ## GRIP @ The Sparks Foundation

# In[1]:


# Importing the required libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  


# ###  Step 1 - Reading the data from source
# 

# In[2]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# ###  Step 2 - Input data Visualization
# 
# #### Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[3]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ##### From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# ###  Step 3- Data Processing (Preparing the data)
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[4]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values  


# #### Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# ### Step 4 -Training the Algorithm
# We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

# In[6]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# ###  Step 5 - Plotting the Line of regression
# Now since our model is trained now, its the time to visualize the best-fit line of regression.

# In[7]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red');
plt.show()


# ###  Step 6 - Making Predictions
# Now that we have trained our algorithm, it's time to make some predictions.
# 
# For this we will use our test-set data

# In[8]:


# Testing data
print(X_test)
# Model Prediction 
y_pred = regressor.predict(X_test)


#  ### Step 7 - Comparing Actual result to the Predicted Model result

# In[9]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df 


# In[10]:


# Solution to given question
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### Step8 -  **Evaluating the model**
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[11]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 

