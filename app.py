import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import pickle


app = Flask("__name__")

df_1=pd.read_csv("Churn_Modelling.csv")

q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    '''
    Credit Score
    Age
    Geography
    gender
    tenure
    Balance
    NumOfProducts
    HasCrCard
    IsActiveMember
    EstimatedSalary
    '''
    
    inputQuery1 = request.form['Credit Score']
    #inputQuery2 = request.form['Geography']
   # inputQuery3 = request.form['Gender']
    inputQuery4 = request.form['Age']
    inputQuery5 = request.form['Tenure']
    inputQuery6 = request.form['Balance']
    inputQuery7 = request.form['NumOfProducts']
    inputQuery8 = request.form['HasCrCard']
    inputQuery9 = request.form['IsActiveMember']
    inputQuery10 = request.form['EstimatedSalary']
    
    model = pickle.load(open("XGB_model.pkl", "rb"))
    
    
    data = [[inputQuery1,inputQuery4, inputQuery5, inputQuery6, inputQuery7, inputQuery8, inputQuery9, inputQuery10]]
    new_df = pd.DataFrame(data, columns = ['Credit Score','Age', 'Tenure', 'Balance',     'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'])
    df_2 = pd.concat([df_1, new_df], ignore_index = True) 
    single = model.predict(df_2.tail(1))
    probablity = model.predict_proba(df_2.tail(1))[:,1]

    if single==1:
        o1 = "The customer is going to churn"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "The customer is not going to churn"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('index.html', output1=o1, output2=o2,
                           query1 = request.form['Credit Score'],
                           query4 = request.form['Age'],
                           query5 = request.form['Tenure'],
                           query6 = request.form['Balance'],
                           query7 = request.form['NumOfProducts'],
                           query8 = request.form['HasCrCard'],
                           query9 = request.form['IsActiveMember'],
                           query10 = request.form['EstimatedSalary'])
    
if __name__ == "__main__":
    app.run(debug=True)