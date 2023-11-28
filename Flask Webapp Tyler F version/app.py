from flask import Flask, render_template, request
import pickle
import numpy as np


#create flask app

app = Flask(__name__)


# load the pickled model

model = pickle.load(open("model.pkl", "rb"))


#create a flask endpoints

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    float_vals = [float(x) for x in request.form.values()]
    vals = [np.array(float_vals)]

    predictions = model.predict(vals)

    return render_template("index.html", prediction = predictions)




if __name__ == '__main__':
    app.run(debug=True)