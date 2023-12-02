from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from io import BytesIO 
import base64 


#create flask app

app = Flask(__name__)


# load the pickled model

model = pickle.load(open("model.pkl", "rb"))


#create a flask endpoints

@app.route('/')
def home():
    return render_template("index.html", prediction=None)

weather_df = pd.read_csv('data/seattle-weather.csv')
@app.route('/visuals') 
def visuals(): 
    # Assuming 'date' is in datetime format, if not, convert it using pd.to_datetime() 
    last_week_data = weather_df.tail(7) 
    
    # Plotting 
    plt.figure(figsize=(10, 6)) 
    plt.plot(last_week_data['date'], last_week_data['weather'], marker='o') 
    plt.title('Weather for the Last 7 Days') 
    plt.xlabel('Date') 
    plt.ylabel('Weather') 
    plt.xticks(rotation=45) 
    
    # Save plot to a BytesIO object 
    img = BytesIO() 
    plt.savefig(img, format='png') 
    img.seek(0) 
    
    # Encode the plot to base64 for embedding in HTML 
    plot_url = base64.b64encode(img.getvalue()).decode() 
    
    return render_template('visuals.html', plot_url=plot_url)

@app.route('/predict', methods=["POST"])
def predict():
    float_vals = [float(x) for x in request.form.values()]
    vals = [np.array(float_vals)]

    predictions = model.predict(vals)

    return render_template("index.html", prediction = predictions)




if __name__ == '__main__':
    app.run(debug=True)