# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 21:11:12 2020
@author: pattn
"""


# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle


app = Flask("__name__")



q = ""

@app.route("/")
def loadPage():
	return render_template('index.html', query="")


@app.route("/predict", methods=['POST'])
def predict():
    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    model = pickle.load(open("XGB_model.pkl", "rb"))
    
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    new_df = pd.DataFrame(df, columns = ['Credit Score', 'Geography','Gender','Age', 'Tenure', 'Balance', 'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'])
    
    single = XGBoost.predict(new_df)
    probablity = XGBoost.predict_proba(new_df)[:,1]
    
    if single==1:
        o1 = "The customer is going to churn"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "The customer is not going to churn"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('index.html', output1=o1, output2=o2, query1 = request.form['query1'], query2 = request.form['query2'],query3 = request.form['query3'],query4 = request.form['query4'],query5 = request.form['query5'],query6 = request.form['query6'], query7 = request.form['query7'],query8 = request.form['query8'],query9 = request.form['query9'],query10 = request.form['query10'])
    
if __name__ == "__main__":
    app.run()