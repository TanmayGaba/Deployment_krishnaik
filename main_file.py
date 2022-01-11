import pandas as pd
from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np

APP = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@APP.route('/')
def home():
    return render_template('index.html')

@APP.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))


if __name__ == "__main__":
    APP.run(debug=True)