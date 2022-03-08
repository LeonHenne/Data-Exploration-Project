from pyexpat import model
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import mlflow

def test_evaluate_model(model_run_id):
    #calling the model loading fuction
    model = load_model(model_run_id)
    mlflow.set_experiment("Model_testing")
    #reading the model parameters
    log_params(model_run_id)
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
    print(accuracy)
    return accuracy, matrix

def log_params(run_id):
    mlflow.log_param("loaded_model",run_id)
    mlflow.log_param("Used_evaluation_dataset", "test_dataset")
    mlflow.log_param("Random_state",42)
    mlflow.log_param("Validationset_size", len(y_test))
    model_parameter_file = "mlruns/0/"+run_id+"/params/"

    with open(model_parameter_file+"Number_of_selected_neighbours","r") as f:
        model_kneighbours = f.read()
    print(model_kneighbours)
    mlflow.log_param("Number_of_selected_neighbours", model_kneighbours)

    with open(model_parameter_file+"Trainset_size","r") as f:
        model_testset_size = f.read()
    print(model_testset_size)
    mlflow.log_param("Trainset_size", model_testset_size)

def load_model(model_uri):
    logged_model = "runs:/"+model_run_id+'/knn_model'
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    return loaded_model

model_run_id = "d51c7a19574240ecb8e03559d9b61a82"

test_df = pd.read_csv("data/Titanic_test_dataset.csv")
x_test=test_df.drop('Survived', axis = 1)
scaler= StandardScaler()
scaler.fit(x_test)
x_test= scaler.transform(x_test)
y_test=test_df['Survived']

accuracy, matrix = test_evaluate_model(model_run_id)