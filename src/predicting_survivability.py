import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mlflow


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

    x_train=df.drop('Survived', axis = 1)
    y_train=df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, train_size=0.8, random_state=42)
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.25,train_size=0.75, random_state=42)
    test_data = pd.concat([x_test, y_test], axis= 1)
    test_data.to_csv("data/Titanic_test_dataset.csv", index=False)
    return x_train,x_validate, x_test, y_train,y_validate, y_test

def feature_preperation(df):
    # transforming feature type sex into 0 and 1 instead of female and male
    df["Sex"] = df["Sex"].map(dict({'male': 1,'female': 0}))
    
    return df

def feature_scaling():
    scaler= StandardScaler()
    scaler.fit(x_train)
    scaled_x_train= scaler.transform(x_train)
    scaled_x_validate = scaler.transform(x_validate)
    scaled_x_test= scaler.transform(x_test)

    return scaled_x_train,scaled_x_validate,scaled_x_test
    
def train_knn(x_train,y_train,knn_param):
    knn_model=KNeighborsClassifier(n_neighbors=knn_param)
    knn_model.fit(x_train, y_train)
    return knn_model

def validation_evaluate_model(model: KNeighborsClassifier):
    mlflow.log_param("Used_evaluation_dataset", "validation_dataset")
    y_predicted = model.predict(x_validate)
    accuracy = accuracy_score(y_validate,y_predicted)
    matrix = confusion_matrix(y_validate,y_predicted)
    print(str(knn_parameter)+" "+ str(accuracy))
    return accuracy, matrix

def test_evaluate_model(model: KNeighborsClassifier):
    mlflow.log_param("Used_evaluation_dataset", "test_dataset")
    y_predicted = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_predicted)
    matrix = confusion_matrix(y_test,y_predicted)
    print(str(knn_parameter) + str(accuracy))
    return accuracy, matrix


df = load_data()
df_cleaned = data_cleaning(df)
df_preped = feature_preperation(df_cleaned)
x_train,x_validate, x_test, y_train,y_validate, y_test= splitting_dataset(df_preped)
x_train,x_validate, x_test = feature_scaling()

for knn_parameter in range(1,2):
    with mlflow.start_run():
        #logging the model parameter
        mlflow.log_param("Random_state", 42)
        mlflow.log_param("Trainset_size", len(x_train))
        mlflow.log_param("Validationset_size",len(x_validate))
        knn_hyperparameter = knn_parameter
        mlflow.log_param("Number_of_selected_neighbours", knn_hyperparameter)

        #training and evaluating the KNN model
        knn_model = train_knn(x_train,y_train,knn_hyperparameter)
        accuracy, matrix = validation_evaluate_model(knn_model)

        # logging evaluation metrics and the model itself
        mlflow.log_metric("Accuracy",accuracy)
        true_positive = matrix[0][0]
        true_negative = matrix[1][1]
        false_positive = matrix[0][1]
        false_negative = matrix[1][0]
        mlflow.log_metric("true_positive", true_positive)
        mlflow.log_metric("true_negative", true_negative)
        mlflow.log_metric("false_positive", false_positive)
        mlflow.log_metric("false_negative", false_negative)
        mlflow.sklearn.log_model(knn_model, "knn_model")

        mlflow.end_run()