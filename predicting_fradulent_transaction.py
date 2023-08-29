#!/usr/bin/env python
# coding: utf-8

# In[5]:


#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[6]:


#loading dataset and running some basic methods to know more about datasets
df = pd.read_csv('./input/dataset/Fraud.csv')


# In[7]:


df.head()


# In[8]:


df.shape


# In[9]:


df.info()


# In[10]:


df.columns


# In[11]:


df[df.duplicated()]


# In[12]:


df.isnull().sum()


# In[13]:


df.describe()


# #checking for outliers using box plot
# for col for df.columns:
#     if df[col].dtype = 'float64' or df[col].dtype = 'int64':
#         print(col)
#         df.boxplot(column = col)
#         plt.show()
#         

# In[14]:


numerical_columns = ['step','amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Initialize a dictionary to store the number of outliers for each column
outliers_count = {}

for col in numerical_columns:
    # Calculate the IQR for each numerical column
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Identify potential outliers using the IQR method
    outliers = ((df[col] < (Q1 - 1.5 * IQR)) |
                (df[col] > (Q3 + 1.5 * IQR)))
    
    # Count the number of outliers for the current column
    num_outliers = outliers.sum()
    
    # Store the count in the dictionary
    outliers_count[col] = num_outliers

# Display the number of outliers for each column
for col, count in outliers_count.items():
    print(f"Number of outliers in column '{col}': {count}")


# In[15]:


sns.countplot(df["isFraud"])
df.isFraud.value_counts()


# In[16]:


sns.countplot(df["isFlaggedFraud"])
df.isFlaggedFraud.value_counts()


# In[17]:


df.corr()


# In[20]:


correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[21]:


# Create new columns for balance changes
df['balanceChangeOrig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Drop the original balance columns
df.drop(['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'], axis=1, inplace=True)


# In[22]:


correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True)
plt.show()


# In[23]:


df.type.value_counts()


# In[24]:


col_name = df.columns.tolist()
print(col_name)


# In[25]:


encoded_types = pd.get_dummies(df['type'], prefix='type')
df = pd.concat([df, encoded_types], axis=1)

# Drop the original 'type' column
df.drop(['type'], axis=1, inplace=True)


# In[26]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['nameOrig'] = labelencoder.fit_transform(df['nameOrig'])
df['nameDest'] = labelencoder.fit_transform(df['nameDest'])


# In[27]:


df


# In[28]:


from sklearn.model_selection import train_test_split

# Splitting the data into features (X) and target (Y)
X = df.drop(['isFraud'], axis=1) # Feature
y= df['isFraud'] # Tagret

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


from sklearn.preprocessing import StandardScaler
# Feature Scaling: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[30]:


from sklearn.linear_model import LogisticRegression
# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)


# In[31]:


model.score(X_train_scaled, y_train)


# In[32]:


model.score(X_test_scaled, y_test)


# In[33]:


from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(random_state = 2)
model_dt.fit(X_train_scaled, y_train)


# In[34]:


feature_importances = model_dt.feature_importances_

# Create a dataframe to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importances
})

# Sort feature by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the feature importance DataFrame
print(feature_importance_df)


# In[ ]:




