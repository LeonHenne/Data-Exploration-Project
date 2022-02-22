import numpy as np
import pandas as pd


def load_data():
    #loading the dataset
    df = pd.read_csv("data/Titanic.csv") # TODO: Gist erstellen
    return df

def explore_data():
    # provides a brief introduction to the structure of the dataset
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.isnull().sum())

def data_cleaning(df):
    #Cleaning the data from null values
    # Deleting the columns "Cabin" due to only 22.9 % non missing values (891 total rows - 678 null values / 891 = 0.2289)
    # Dropping columns like Embarked, Ticket, Name and Fare due to irrelevant information for predicting the survivability
    df.drop(["Cabin", "Embarked", "Ticket","Fare", "Name"],axis=1,inplace=True)
    # Replace null values in the Age column with the median age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    print(df.isnull().sum())
    return df

# def feature_preperation(df):

df = load_data()
df = data_cleaning(df)
# df = feature_preperation(df)
print(df.shape)

