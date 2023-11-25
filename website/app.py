import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create Flask app
app = Flask(__name__)

# Load seven models and store them in a dictionary
models = {}
pkl_files = ['germanBusiness_model.pkl', 'germanCarNew_model.pkl', 'germanCarUsed_model.pkl',
            'germanFurniture_model.pkl', 'germanRadio_TV_model.pkl', 'germanRepairs_model.pkl']
for file in pkl_files:
    model_name = file[:-10]
    models[model_name] = pickle.load(open(file, "rb"))

# Routes
@app.route("/")
def home():
    return render_template("index.html", models=models.keys())

@app.route("/predict", methods=["POST"])
def predict():
    selected_model_name = request.form.get("selected_model")
    selected_model = models[selected_model_name]

    #int_features = [int(x) for x in request.form.values() if isinstance(x, (int, str))]
    #features = [np.array(int_features)]
    int_features = []
	
    for key in request.form.keys():
        if key != 'selected_model':
            int_features.append(int(request.form[key]))
                        
    features = [np.array(int_features)]

    prediction = selected_model.predict(features)

    output = "HIGH CREDIT RISK"
    if prediction == 1:
        output = "LOW CREDIT RISK"

    return render_template("index.html", models=models.keys(), selected_model=selected_model_name,
                           prediction_text=f"The prediction for {selected_model_name} is {output}")

if __name__ == "__main__":
    app.run(debug=True)
