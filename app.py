import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("RandomForestClassifier_NEW.pkl", "rb"))

def convert_to_str(word):
    word_dict = {0: 'died', 1: 'lived'}
    return word_dict[word]

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    output = convert_to_str(prediction[0])
    return render_template("index.html", prediction_text = "The Horse is {}".format(output))

if __name__ == "__main__":
    flask_app.run(debug=True)