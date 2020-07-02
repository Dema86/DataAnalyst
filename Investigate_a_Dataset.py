#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Project: No-Show appointments a Dataset From Kaggle 
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > The data set I have selected is Medical Appointment No Shows . The reason why I have selected this dataset is becuase as I am working for a healtcare firm, I would like to understand patient behaviour or characteritics behind on why does a patient show up or do not show up for appointment.
# 
# The dataset variables are self explanatory and the orginal variables that exist here are 'PatientId', 'AppointmentID', 'Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'No-show'
# 

# In[4]:



# Use this cell to set up import statements for all of the packages that you
#   plan to use.

# Remember to include a 'magic word' so that your visualizations are plotted
#   inline with the notebook. See this page for more:
#   http://ipython.readthedocs.io/en/stable/interactive/magics.html
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import pandas as pd
import numpy as np                                                   # for linear algebra
import datetime                                                      # to deal with date and time
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# 
# 
# ### General Properties

# In[5]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head(3)


# In[6]:


# Create the summary report ( Find any anomalies in the data)
df.describe()


# In[7]:


df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')
df.head()


# <a id='eda'></a>
# ## Variable Identification
# 
# Minimum age is showing -ve (May be an outlier) and max value of age are ranging above 100
# 
# 
# First we will identify the predictor and target variable

# In[8]:


df.columns


# In[9]:


df.dtypes


# In[10]:


#Check missing values
df.isnull().sum()


# In[11]:


#structure of the data
df.info()


# In[12]:


df['No-show'].value_counts()


# In[13]:


df['Gender'].value_counts()


# In[14]:


df['Gender'].value_counts()


# In[15]:


#Check for scheduled day and appointment day  
df[['ScheduledDay','AppointmentDay']].head(5)


# In[16]:


# # Convert that dateformat which is in string to datetime64[ns]
# The day of the week with Monday=0, Sunday=6

df['ScheduledDay'] = df['ScheduledDay'].apply(np.datetime64)
df['Day_Scheduled'] = df['ScheduledDay'].dt.day
df['weekday_Scheduled'] = df['ScheduledDay'].dt.dayofweek
df['Month_Scheduled'] = df['ScheduledDay'].dt.month


df['AppointmentDay'] = df['AppointmentDay'].apply(np.datetime64)
df['Day_appointed'] = df['AppointmentDay'].dt.day
df['weekday_appointed'] = df['AppointmentDay'].dt.dayofweek
df['Month_appointed'] = df['AppointmentDay'].dt.month


# In[17]:


df.head(2)


# In[18]:


df.columns


# In[19]:


df['Month_appointed'].nunique()


# In[20]:


#Rename the columns which have incorrect spelling mistakes - this will helps us create columns in easy to understand way
df.rename(columns = {'Hipertension' : 'Hypertension', 'Handcap':'Handicap', 'No-show' : 'NoShow'}, inplace = True)
df.head(3)


# In[21]:


# find the unique values for each of the columns specified
print("the unique values for 'Gender' are {}".format(df.Gender.unique()))
print("the unique values for 'Age' are {}".format(sorted(df.Age.unique())))
print("the unique values for 'Neighbourhood' are {}".format(df.Neighbourhood.unique()))
print("the unique values for 'Scholarship' are {}".format(df.Scholarship.unique()))
print("the unique values for 'Hypertension' are {}".format(df.Hypertension.unique()))
print("the unique values for 'Diabetes' are {}".format(df.Diabetes.unique()))
print("the unique values for 'Alcoholism' are {}".format(df.Alcoholism.unique()))

print("the unique values for 'Handicap' are {}".format(df.Handicap.unique()))
print("the unique values for 'SMS_received' are {}".format(df.SMS_received.unique()))
print("the unique values for 'NoShow' are {}".format(df.NoShow.unique()))


# In[22]:


# Check how many records with age < 0 and age > 100
df.query('Age < 0  | Age > 100').count()


# In[23]:


df.drop(df[(df.Age < 0) | (df.Age > 100)].index, inplace = True)


# In[24]:


df.info()


# In[25]:


#Uniuqe patient counts in the dataset
df.PatientId.nunique()


# ### Univariate Analysis
# We will first look at the target variable, i.e., NoShow. As it is a categorical variable, let us look at its frequency table, percentage distribution and bar plot. Frequency table of a variable will give us the count of each category in that variable.

# In[26]:


df['NoShow'].value_counts()


# In[27]:


# Normalise can be set to true to print the proportions instead of Numbers.
df['NoShow'].value_counts(normalize=True)


# In[28]:


df['NoShow'].value_counts().plot.bar(figsize = (4,4), title = 'NoShow dataset', color = 'g')
plt.xlabel('NoShow')
plt.ylabel('Count');


# In[29]:


plt.figure(1)
plt.figure(figsize = (15,10))
iter_cols = ['Gender', 'Scholarship', 'Hypertension', 'Diabetes']
#              , 'Alcoholism', 'Handicap']
colour = ['c', 'g', 'b', 'r']
i = 0
for col in iter_cols:    
    plt.subplot(int(str(22)+str((iter_cols.index(col)+1))))
    df[col].value_counts(normalize=True).plot.bar(figsize=(20,10), fontsize = 15.0, color = colour[i])
    plt.title(col, fontweight="bold", fontsize = 22.0)
    plt.ylabel('Count %', fontsize = 20.0)
    i = i +1

It can be inferred from the above bar plots that:

~65% patient instances are with female records
Around 85% of the these instances are not having scholarship.
~20% of patient instances have Hypertension.
~88% of patient instacnes do not have diabetes
# # Exploratory Data Analysis
# Now I will be answering all the questions mentioned above with my analysis
# 
# 1. Is there any Gender difference in having a patient to be with status Show / No-Show?

# In[30]:


fig, ax = plt.subplots()
Gender=pd.crosstab(df['Gender'],df['NoShow'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(9,4),  ax = ax);
ax.set_facecolor('#d8dcd6')


# <a id='conclusions'></a>
# ## Conclusions
# 
# 1.Of those patients who have showed up for appointment, majority patients with age group from 20 to 40 did not show up for the appointment when compared to age groups 0-20, 40-60 and 60 plus.of these 20 to 40 age group, Female patients are NOT active in getting appointment with doctors when compared to similar age group of Males
# 
# 2.Majority of the visits or appointments happend on the weeekdays (Monday, Tuesday and Wednesday) when compared to weekends. 
# 
# 3.Hence weekdays are the best for good conversion for appointment Hypertension and Diabetes patients are attending the doctor when compared to other medications. Age ranges from 40 to 80 - with females conversion is better than Males.
# 
# 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

