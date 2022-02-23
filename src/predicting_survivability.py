from re import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score



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
    df.drop(["Cabin", "Embarked", "Ticket","Fare", "Name", "PassengerId"],axis=1,inplace=True)
    # Replace null values in the Age column with the median age
    df['Age'] = df['Age'].fillna(df['Age'].median())
    print(df.isnull().sum())
    return df

def splitting_dataset(df : pd.DataFrame):

    X_train=df.drop('Survived', axis = 1)
    y_train=df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, train_size=0.8, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.25,train_size=0.75, random_state=42)

    return X_train,X_validate, X_test, y_train,y_validate, y_test

def feature_preperation(df):
    # transforming feature types TODO: sex into 0 and 1 / ( age in range between 0 and 1 )
    df["Sex"] = df["Sex"].map(dict({'male': 1,'female': 0}))
    return df
    
def train_knn(X_train,y_train,knn_param):
    knn_model=KNeighborsClassifier(n_neighbors=knn_param)
    knn_model.fit(X_train, y_train)
    return knn_model

def evaluate_model(model: KNeighborsClassifier,X_validate, y_validate):
    y_predicted = knn_model.predict(X_validate)
    accuracy = accuracy_score(y_validate,y_predicted)
    print(accuracy)
    return accuracy

df = load_data()
df_cleaned = data_cleaning(df)
df_preped = feature_preperation(df_cleaned)
X_train,X_validate, X_test, y_train,y_validate, y_test= splitting_dataset(df)

knn_model = train_knn(X_train,y_train,1)
accuracy = evaluate_model(knn_model,X_validate, y_validate)
