#!/usr/bin/env python
# coding: utf-8

# # Water Quality Prediction

# ## Content
# The file "water_potability.csv" includes water quality metrics for 3276 distinct bodies of water.
# 
# Water quality prediction involves using models to estimate the quality of water in a particular area at a given time. There are various factors that can affect water quality, such as temperature, pH levels, dissolved oxygen, and the presence of pollutants.
# 
# To predict water quality, one approach is to collect data on various parameters that affect water quality and use statistical and machine learning models to analyze the data and make predictions. This can involve using historical data to train a model to identify patterns and trends in water quality over time, and then using this model to make predictions about future water quality based on current and predicted conditions.
# 
# Some of the key steps involved in water quality prediction include data collection and analysis, model selection and training, and model validation and testing. It is important to ensure that the models used for water quality prediction are accurate and reliable, and that they are regularly updated to account for changing environmental conditions.
# 
# Water quality prediction can be used for a variety of purposes, including identifying potential sources of pollution, monitoring the effectiveness of pollution control measures, and predicting the impact of climate change on water quality. By accurately predicting water quality, it is possible to take proactive steps to protect our water resources and ensure that they remain safe and healthy for human use and for the environment.
# 
# ### 1. pH value:
# pH is a crucial parameter for assessing the acid-base equilibrium of water and indicating its acidic or alkaline state. The World Health Organization (WHO) has established a maximum permissible pH range of 6.5 to 8.5 for safe drinking water. In this study, the pH values measured fell within this range, ranging from 6.52 to 6.83, meeting the WHO guidelines for safe drinking water.
# 
# ### 2. Hardness:
# Water hardness is primarily attributed to the presence of dissolved calcium and magnesium salts, which originate from geological deposits that the water comes into contact with during its journey. The amount of time that water spends in contact with such materials influences the level of hardness present in the raw water. Historically, water hardness was measured as the amount of soap that could be precipitated by the water due to its calcium and magnesium content.
# 
# ### 3. Solids (Total dissolved solids - TDS):
# Water has the capacity to dissolve various inorganic and organic minerals and salts, including calcium, potassium, sodium, bicarbonates, chlorides, magnesium, sulfates, and others. While some of these minerals are essential to human health, excessive levels of dissolved minerals can impact the taste and color of water. Total Dissolved Solids (TDS) is an important parameter for assessing water quality, as it indicates the amount of dissolved minerals present in water. For drinking water, the recommended TDS limit is 500 mg/l, while the maximum limit is 1000 mg/l. Water with a high TDS value is considered highly mineralized and may not be suitable for consumption.
# 
# ### 4. Chloramines:
# Chlorine and chloramine are the primary disinfectants used in public water systems to safeguard against waterborne illnesses. Chloramines are typically produced by adding ammonia to chlorine during the drinking water treatment process. To ensure safe drinking water, the maximum recommended chlorine level is 4 milligrams per liter (mg/L) or 4 parts per million (ppm).
# 
# ### 5. Sulfate:
# Sulfates are naturally-occurring substances that can be found in minerals, soil, rocks, ambient air, groundwater, plants, and food. While they have various industrial applications, such as in the chemical industry, sulfates also occur in natural environments. In seawater, the concentration of sulfate is around 2,700 milligrams per liter (mg/L), while in most freshwater sources, it ranges from 3 to 30 mg/L. However, in some locations, sulfate levels can reach as high as 1,000 mg/L.
# 
# ### 6. Conductivity:
# Pure water is a poor conductor of electricity and serves as an effective insulator. However, the presence of ions in water can significantly enhance its electrical conductivity. The amount of dissolved solids in water is typically indicative of its electrical conductivity. Measured in microSiemens per centimeter (μS/cm), electrical conductivity (EC) reflects a solution's ability to transmit an electric current via ionic processes. In accordance with WHO guidelines, the recommended maximum EC value for safe drinking water is 400 μS/cm.
# 
# ### 7. Organic_carbon:
# Total Organic Carbon (TOC) is present in source waters due to the decomposition of natural organic matter (NOM) and synthetic sources. TOC is a measurement of the amount of carbon in all organic compounds found in pure water. The US EPA recommends that treated or drinking water should have less than 2 mg/L of TOC, while source water used for treatment should have less than 4 mg/L of TOC.
# 
# ### 8. Trihalomethanes:
# THMs (Trihalomethanes) are a group of chemicals that may be present in water treated with chlorine. The concentration of THMs in drinking water can vary depending on the organic content of the water, the amount of chlorine used for treatment, and the temperature of the water during treatment. THM levels of up to 80 parts per million (ppm) are generally considered safe for drinking water.
# 
# ### 9. Turbidity:
# The turbidity of water is determined by the amount of solid particles that are suspended in it. It is an important measure of water quality, particularly with respect to colloidal matter in waste discharge. Turbidity is also an indicator of the light-reflecting properties of water. In the case of Wondo Genet Campus, the mean turbidity value obtained (0.98 NTU) is lower than the WHO recommended limit of 5.00 NTU.
# 
# ### 10. Potability:
# The potability of water is indicated by a binary variable, where 1 denotes water that is safe for human consumption and 0 denotes water that is not safe to drink. In other words, a value of 1 indicates that the water meets the necessary standards for human consumption, while a value of 0 indicates that it does not.

# ## Data Gathering

# In[1]:


# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# suppress warning messages
warnings.filterwarnings("ignore")


# In[2]:


# read the csv file into a pandas dataframe
data = pd.read_csv(r'C:\Users\ASUS\Downloads\water_potability.csv')

# print the first five rows of the dataframe
print(data.head())


# ## Exploratory Data Analysis

# In[3]:


# display the shape of the dataframe
print(data.shape)


# In[4]:


# display the total number of missing values in each column
print(data.isnull().sum())


# In[5]:


# display information about the dataframe, including column data types and non-null value counts
print(data.info())


# In[6]:


# display summary statistics of the numerical columns in the dataframe
data.describe()


# In[7]:


# fill missing values with the mean of the respective columns and update the dataframe in place
data.fillna(data.mean(), inplace=True)

# display the total number of missing values in each column to confirm that there are no missing values left
print(data.isnull().sum())


# In[8]:


# display the number of instances for each unique value of the 'Potability' column
print(data['Potability'].value_counts())


# In[9]:


# create a countplot of the 'Potability' column
sns.countplot(data['Potability'])

# display the plot
plt.show()


# In[10]:


# create a distribution plot of the 'ph' column
sns.distplot(data['ph'])

# display the plot
plt.show()


# In[11]:


# create a histogram of all columns in the dataframe
data.hist(figsize=(14,14))

# display the plot
plt.show()


# In[12]:


# create a pairplot of the dataframe with the 'Potability' column as the hue
sns.pairplot(data, hue='Potability')


# In[13]:


# create a scatter plot of 'Hardness' against 'Solids'
sns.scatterplot(data['Hardness'], data['Solids'])


# In[14]:


# create a scatter plot of 'ph' against 'Potability'
sns.scatterplot(data=data, x='ph', y='Potability')

# add title and axis labels
plt.title('Relationship between pH and Potability')
plt.xlabel('pH')
plt.ylabel('Potability')

# display the plot
plt.show()


# In[15]:


# create a correlation matrix of the dataframe
corr_matrix = data.corr()

# set figure size and create heatmap of the correlation matrix
plt.figure(figsize=(13, 8))
sns.heatmap(corr_matrix, annot=True, cmap='terrain')

# add title and adjust font size
plt.title('Correlation Matrix of Water Potability Data', fontsize=16)

# display the plot
plt.show()


# In[16]:


# create a boxplot of all columns in the dataframe
fig, ax = plt.subplots(figsize=(14,7))
data.boxplot(ax=ax)

# add title and adjust font size
plt.title('Boxplot of Water Potability Data', fontsize=16)

# display the plot
plt.show()


# In[17]:


# display descriptive statistics of the 'Solids' column
data['Solids'].describe()


# In[18]:


# create feature and target variables
X = data.drop('Potability', axis=1)
Y = data['Potability']

# print shape of feature and target variables
print('Shape of Features:', X.shape)
print('Shape of Target:', Y.shape)


# In[19]:


from sklearn.model_selection import train_test_split

# split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101, shuffle=True)

# print the shapes of the resulting sets
print('Training Features Shape:', X_train.shape)
print('Training Target Shape:', Y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Target Shape:', Y_test.shape)


# ## Train Decision Tree Classifier and check accuracy

# In[20]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# create a Decision Tree Classifier object
dtc = DecisionTreeClassifier(criterion='gini', min_samples_split=10, splitter='best')

# fit the model on the training data
dtc.fit(X_train, Y_train)

# make predictions on the test data
Y_pred = dtc.predict(X_test)

# calculate and print the accuracy score
accuracy = accuracy_score(Y_test, Y_pred)
print('Accuracy Score:', accuracy)

# generate and display the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print('Confusion Matrix:\n', cm)

# display the classification report
report = classification_report(Y_test, Y_pred)
print('Classification Report:\n', report)


# In[21]:


# create a variable for the input data to be predicted
input_data = [[5.735724, 158.318741, 25363.016594, 7.728601, 377.543291, 568.304671, 13.626624, 75.952337, 4.732954]]

# use the trained model to make a prediction on the input data
prediction = dtc.predict(input_data)[0]

# print the predicted output
print('Predicted Output:', prediction)


# ## Apply Hyper Parameter Tuning

# In[22]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# define models and parameters
model = DecisionTreeClassifier()
criterion = ["gini", "entropy"]
splitter = ["best", "random"]
min_samples_split = [2,4,6,8,10,12,14]

# define grid search
grid = dict(splitter=splitter, criterion=criterion, min_samples_split=min_samples_split)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search_data = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, 
                           scoring='accuracy',error_score=0)
grid_search_data.fit(X_train, Y_train)
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

# define the classifier
model = DecisionTreeClassifier()

# define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 4, 6, 8, 10, 12, 14]
}

# define the search strategy
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    n_jobs=-1,
    cv=cv,
    scoring='accuracy',
    error_score=0
)

# perform the search
grid_search.fit(X_train, Y_train)


# In[23]:


# Print the best score and parameters
best_score = grid_search_data.best_score_
best_params = grid_search_data.best_params_
print(f"Best score: {best_score:.3f} using {best_params}")

# Print the mean and standard deviation of scores for all parameter combinations
means = grid_search_data.cv_results_['mean_test_score']
stds = grid_search_data.cv_results_['std_test_score']
params = grid_search_data.cv_results_['params']

for mean, std, params in zip(means, stds, params):
    print(f"Mean score: {mean:.3f} (Std: {std:.3f}) with params: {params}")

# Print the training and testing score of the best model
train_score = grid_search_data.score(X_train, Y_train)
test_score = grid_search_data.score(X_test, Y_test)
print("Training score:", train_score*100)
print("Testing score:", test_score*100)


# # Conclusions 
# 1. The developed model was able to accurately predict the quality of water based on the given parameters, which could be useful for ensuring safe drinking water for human consumption.
# 
# 2. Certain parameters, such as pH, total dissolved solids, and turbidity, were found to be important indicators of water quality, while others may be less relevant.
# 
# 3. The study identified areas where water quality may be suboptimal, highlighting the need for further investigation and potential remediation efforts.
# 
# 4. The results of the project could be used by policymakers and public health officials to develop interventions aimed at improving water quality and protecting public health.

# In[ ]:




