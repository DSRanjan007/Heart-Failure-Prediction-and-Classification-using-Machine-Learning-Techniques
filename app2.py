# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 14:07:50 2021

@author: RANJAN_KESHRI
"""
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Heart_Stroke_prediction_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    if request.method == 'POST':
    
       
        predicted_age_of_marriage = model.predict([[
                            float(request.form['age']),
                            float(request.form['anameia']),
                            float(request.form['creatinine_phosphokinase']),
                            float(request.form['diabetes']),
                            float(request.form['ejection_fraction']),
                            float(request.form['high_blood_pressure']),
                            float(request.form['platelets']),
                            float(request.form['serum_creatinine']),
                            float(request.form['serum_sodium']),
                            float(request.form['sex']),
                            float(request.form['smoking']),
                            float(request.form['time'])
                                 
                           ]])
        output=round(predicted_age_of_marriage[0],2)
        
        if output==0:
            return render_template('index.html',prediction_text="Low Chance of Heart Failure")
        else:
            return render_template('index.html',prediction_text="High Chance of Heart Failure")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
