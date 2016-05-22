
# coding: utf-8

# In[1]:

#Import necessary libraries
import pandas as pd
import numpy as np

def cleanup(csv_name):
    #Open up the csv file
    df = pd.read_csv(csv_name,header=0)
    # header=0 means that row 0 is its header
    # mapping female, male into integers
    df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(int)

    # Dealing with null values by taking the median as their values
    #1.Create an alternative column for age
    df['AgeFill']=df['Age']
    # Taking the median for each gender and pclass
    med_ages = np.zeros((2,3))
    for i in range(2):
        for j in range(3):
            med_ages[i,j]=df[(df.Gender==i)&(df.Pclass==j+1)]['Age'].    dropna().median()
    for i in range(2):
        for j in range(3):
            df.loc[(df.Gender==i)&(df.Pclass==j+1)&(df['Age'].isnull()),               'AgeFill']=med_ages[i,j]

    # Creating a column that tells us whether the Age column 
    # was originally null or not
    df['AgeIsNull']=pd.isnull(df.Age).astype(int)

    # Extract columns with string values
    string_columns = df.columns[df.dtypes.map(lambda x: x=='object')]
    string_columns = string_columns.values

    # drop string columns
    df_cleaned = df.drop(string_columns, axis=1)
    #Delete the age column, since it is needless
    df_cleaned.pop('Age')
    #Drop NA columns
    df_cleaned = df_cleaned.dropna()
    return df_cleaned