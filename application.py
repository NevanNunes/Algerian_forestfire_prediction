from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
application = Flask(__name__)
app=application

ridge_model = pickle.load(open('models/model.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])

def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('temperature'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        DC = float(request.form.get('dc'))
        ISI = float(request.form.get('isi'))
        BUI = float(request.form.get('bui'))
        classes = request.form.get('classes')
        region = request.form.get('region')

        new_data_scaled=standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI]])
        result=ridge_model.predict(new_data_scaled)
        # pass the template name and the result variable correctly to render_template
        return render_template('home.html', result=result[0])

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
