import flask
from flask import request
from flask import Flask, render_template, request

import requests
import pickle
import json

from sklearn.preprocessing import StandardScaler
app = flask.Flask(__name__)
app.config["DEBUG"] = True

from flask_cors import CORS
CORS(app)

# main index page route
@app.route('/')
def home():
    return '<h1>API is working.. </h1>'


@app.route('/predict',methods=['GET'])
def predict():
    import pickle
    model = pickle.load(open('Heart_Stroke_prediction_model.pkl', 'rb'))
    
    predicted_age_of_marriage = model.predict([[
                            float(request.args['age']),
                            float(request.args['anameia']),
                            float(request.args['creatinine_phosphokinase']),
                            float(request.args['diabetes']),
                            float(request.args['ejection_fraction']),
                            float(request.args['high_blood_pressure']),
                            float(request.args['platelets']),
                            float(request.args['serum_creatinine']),
                            float(request.args['serum_sodium']),
                            float(request.args['sex']),
                            float(request.args['smoking']),
                            float(request.args['time'])
                                 
                           ]])
    
    return str(round(predicted_age_of_marriage[0],2))


if __name__ == "__main__":
    app.run(debug=True)