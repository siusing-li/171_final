import numpy as np
from flask import Flask, request, render_template
import pickle


# Load six logistic regression models and store them in a dictionary
log_models = {}
log_pkl_files = ['Log_germanBusiness.pkl', 'Log_germanCarNew.pkl', 'Log_germanCarUsed.pkl',
            'Log_germanFurniture.pkl', 'Log_germanRadio_TV.pkl', 'Log_germanRepairs.pkl',
            'Log_germanAppliance.pkl', 'Log_germanOther.pkl', 'Log_germanRetraining.pkl', 'Log_germanEducation.pkl']
for file in log_pkl_files:
    model_name = file[10:-4]
    log_models[model_name] = pickle.load(open(file, "rb"))

# Load six random forest models and store them in a dictionary
rf_models = {}
rf_pkl_files = ['Rf_germanBusiness.pkl', 'Rf_germanCarNew.pkl', 'Rf_germanCarUsed.pkl',
            'Rf_germanFurniture.pkl', 'Rf_germanRadio_TV.pkl', 'Rf_germanRepairs.pkl',
            'Rf_germanAppliance.pkl', 'Rf_germanOther.pkl', 'Rf_germanRetraining.pkl', 'Rf_germanEducation.pkl']
for file in rf_pkl_files:
    model_name = file[3:-4]
    rf_models[model_name] = pickle.load(open(file, "rb"))

# Load six neural network models and store them in a dictionary
nn_models = {}
nn_pkl_files = ['Nn_germanBusiness.pkl', 'Nn_germanCarNew.pkl', 'Nn_germanCarUsed.pkl',
            'Nn_germanFurniture.pkl', 'Nn_germanRadio_TV.pkl', 'Nn_germanRepairs.pkl',
            'Nn_germanAppliance.pkl', 'Nn_germanOther.pkl', 'Nn_germanRetraining.pkl', 'Nn_germanEducation.pkl']
for file in nn_pkl_files:
    model_name = file[3:-4]
    nn_models[model_name] = pickle.load(open(file, "rb"))


# Create Flask app
app = Flask(__name__)

# ROUTES
# Home
@app.route("/")
def home():
    return render_template("index.html", models=log_models.keys())

# Predict
@app.route("/predict", methods=["POST"])
def predict():
    # Get the user choice of model and classifier type
    selected_model_name = request.form.get("selected_model")
    selected_cls = request.form.get("selected_cls")

    # Get the appropriate model pkl file based off user input
    if(selected_cls == "Logistic Regression"):
        selected_model = log_models[selected_model_name]
    elif(selected_cls == "Random Forest"):
        selected_model = rf_models[selected_model_name]
    else:
        selected_model = nn_models[selected_model_name]

    # Get the features from user input
    int_features = []
    for key in request.form.keys():
        if request.form[key] == "Male div":
            int_features.extend([1, 0, 0, 0])
        elif request.form[key] == "Female div/mar":
            int_features.extend([0, 1, 0, 0])
        elif request.form[key] == "Male sing":
            int_features.extend([0, 0, 1, 0])
        elif request.form[key] == "Male mar/wid":
            int_features.extend([0, 0, 0, 1])
        elif key != 'selected_model' and key != 'selected_cls':
            int_features.append(int(request.form[key]))                   
    features = [np.array(int_features)]

    # Make prediction using the model and features
    prediction = selected_model.predict(features)

    # Render the index.html file with provided models and output predictions (Custom good/bad output based on prediction result)
    if(prediction == 1):
        return render_template("index.html", models=log_models.keys(), features = features, good_prediction_text=f"The prediction for this {selected_model_name} loan using {selected_cls} is LOW CREDIT RISK")
    else:
        return render_template("index.html", models=log_models.keys(), features = features, bad_prediction_text=f"The prediction for this {selected_model_name} loan using {selected_cls} is HIGH CREDIT RISK")

# Run the Flask App in main function
if __name__ == "__main__":
    app.run(debug=True)