# 171_final

## To Run the App
1. Navigate to the *website* directory
2. Run *app.py* (type in python app.py to terminal or run from VSCode)
3. Navigate to *127.0.0.1:5000* (type into browser, or click on link from terminal output)
4. Input the specified model, classifier, and features and click *Predict* to see the prediction!
4. For more details, check out this video demo: https://youtu.be/jjfwArwuAMc

## Overview of Repository
### Data Processing and Datasets
The *germanProcessor* jupyter notebook under data_processing folder reads in the original dataset from *171_final/datasets/statlog+german+credit+data/german.data*.  We then process and clean the data using label encoding and one-hot encoding where appropriate.  This is outputted to *germanProcessed.csv*.  We then rename the Purpose attribute to human readable format and this dataset is outputted to *germanProcessed2*.csv.  The jupyter notebook also prints out the heatmap and pairplot for the overall dataset.  We then split the dataset into 10 smaller datasets based off the attribute Purpose and these are outputted to the csvs under *171_final/datasets/cleanData*.  The *germanProcessor* jupyter notebook also contains heatmaps and pairplots for each of the smaller datasets.  Some datasets were additionally modified by dropping unchanging columns, such as *germanAppliance2.csv*.

### Report
You will find the pdf and latex source of our final report here.

### Src
This contains our three machine learning models: Logistic Regression, Random Forest and Nerual Network.  The *germanLogistic* and *germanRandomForest* are implemented using the csvs under *cleanData* folder.  For the neural network, the *cleanData* csvs were too small for the model so it is implemented using the entire dataset (data was still processed using label and onehot encoding).  The *german3HLayerNN(worse)* was our first attempt and the *germanNeuralNetwork* is our final neural network model.  The results of the Logistic Regression and Random Forest models can be seen in their respective jupyter notebooks in the form of classification reports and ROC curves.  The results of the neural network can be found in the *ANN-results.txt* file.

### Website
#### ML Model (model.py)
The best version of each of the three ML models are contained in *model.py*.  This file fits each of the smaller datasets onto each of the models (10 datasets with 3 models each means 30 total) and outputs them to pickle files.  So for example, *Log_germanAppliance.pkl* is the pickle file containing the data for the Logistic Regression model fitted to the germanAppliance dataset. 
#### Flask App (app.py)
Then, *app.py* contains our Flask app which consists of the home (/) and predict (/predict) routes. First, the app reads in the pickle files.  Then, the home route just shows the *index.html* template.  Then, the predict route takes user input from the form contained in *index.html*, makes a prediction using the selected classifier and model based off the user inputted features, and outputs the prediction back to *index.html*.
#### Frontend (index.html and index.css)
Our frontend consists of the *index.html* file under *171_final/website/templates/* and the *index.css* file under *171_final/website/static/styles/*.  The html contains a div for the outputted prediction text (empty on home route, contains prediction on predict route) and a form which takes in user input for the selected classifer, model, and features.  The file also contains a script (triggered by Generate Random Input button) to randomly generate the features within specified ranges.  The file ends off with a submit button to submit the form to the Flask app so that the prediction can be made.
