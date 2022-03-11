import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mlflow

def test_evaluate_model(model_run_id : str) -> Tuple[float,np.ndarray]:
    '''evaluating a given model on test data. Printing hyperparameter and accuracy'''
    #calling the model loading fuction
    model = load_model(model_run_id)

    # setting the experiment to store the test runs separately
    mlflow.set_experiment("Model_testing")
    
    #reading the model parameters
    knn_hyperparameter = log_params(model_run_id)
    #using the loaded model to predict the test data
    y_predicted = model.predict(x_test)

    #calculating the metrices
    accuracy = accuracy_score(y_test,y_predicted)
    matrix = confusion_matrix(y_test,y_predicted)

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
    mlflow.sklearn.log_model(model, "knn_model")

    print(str(knn_hyperparameter) +" "+str(accuracy))

    return accuracy, matrix

def log_params(run_id : str):
    '''logging different parameters of the model performance run on test data'''

    mlflow.log_param("loaded_model",run_id)
    mlflow.log_param("Used_evaluation_dataset", "test_dataset")
    mlflow.log_param("Random_state",42)
    mlflow.log_param("Validationset_size", len(y_test))
    
    # creating the path to the param folder of a given run
    model_parameter_file = "mlruns/0/"+run_id+"/params/"

    with open(model_parameter_file+"Number_of_selected_neighbours","r") as f:
        model_kneighbours = f.read()
    
    mlflow.log_param("Number_of_selected_neighbours", model_kneighbours)

    with open(model_parameter_file+"Trainset_size","r") as f:
        model_testset_size = f.read()

    mlflow.log_param("Trainset_size", model_testset_size)

    return model_kneighbours

def load_model(model_run_id: str) -> mlflow.pyfunc.PyFuncModel:
    '''loading a model from the mlruns folder, given a model_run_id'''

    logged_model = "runs:/"+model_run_id+'/knn_model'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model

def scaling_data(test_df : pd.DataFrame) -> pd.DataFrame:
    '''preparing and scaling the test data'''

    x_test=test_df.drop('Survived', axis = 1)
    #using the sklearn standardscaler
    scaler= StandardScaler()
    scaler.fit(x_test)
    x_test= scaler.transform(x_test)
    y_test=test_df['Survived']
    return x_test,y_test

test_df = pd.read_csv("https://gist.githubusercontent.com/Inwernos/718cbf951b7c5bfc7060801824e64a3c/raw/42fec3053c3532fd521403302bc9e00714e3a26d/Titanic_test_dataset.csv")
x_test, y_test = scaling_data(test_df)
# run_id was figured out through hyperparameter testing
model_run_id = "d51c7a19574240ecb8e03559d9b61a82"
accuracy, matrix = test_evaluate_model(model_run_id)