import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import pickle


#Need to take out germanAppliance, germanOther, germanRetraining (the datasets are too small)
file_list = ['germanBusiness.csv', 'germanCarNew.csv', 'germanCarUsed.csv',
            'germanFurniture.csv', 'germanRadio_TV.csv', 'germanRepairs.csv',
            'germanEducation.csv']

# Initialize an empty dictionary to store DataFrames
dataframes = {}

# Loop through the files and read them into DataFrames
for file in file_list:
    df = pd.read_csv(f"../datasets/cleanData/{file}")
    dataframes[file[:-4]] = df

# Perform the 3 models on each dataset
for name, df in dataframes.items():
    # Splitting training and testing data
    train, test = train_test_split(df, test_size=0.20, random_state=12)
    # The remaining
    X_train, y_train = train.drop(columns=['Class']) ,train['Class']
    X_test, y_test = test.drop(columns=['Class']), test['Class']

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test= sc.transform(X_test)


    # FOR LOGISTIC REGRESSION
    # Define the parameter grid for grid search
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'fit_intercept': [True, False],
        'tol': [1e-3, 1e-4, 1e-5],
    }

    # Initialize the Logistic Regression model
    cls = LogisticRegression(max_iter=1000000)
    #cls.fit(X_train, y_train)

    # Create a GridSearchCV object
    grid_search = GridSearchCV(cls, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_

    # Get the best model from the grid search
    best_log_cls = grid_search.best_estimator_
    #END LOGISTIC REGRESSION


    # FOR RANDOM FOREST
    # Define the hyperparameters and values to tune
    param_grid = {
        'n_estimators': [50, 100, 150, 300],
        'max_depth': [None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 3],
        'criterion': ['gini']
    }

    # Initialize the Random Forest model
    rf_cls = RandomForestClassifier(random_state=24)

    # Perform Grid Search Cross-Validation
    grid_search = GridSearchCV(estimator=rf_cls, param_grid=param_grid, scoring='accuracy', cv=3)
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Get the best model from the grid search
    best_rf_cls = grid_search.best_estimator_
    #END RANDOM FOREST


    # FOR NEURAL NETWORK
    best_nn_cls = MLPClassifier(activation='logistic', hidden_layer_sizes=(3, 15),
              learning_rate_init=0.3, max_iter=500, random_state=42,
              solver='sgd')
    
    best_nn_cls.fit(X_train, y_train)
    #END NEURAL NETWORK


    # Make pickle files of our 3 models
    pickle.dump(best_log_cls, open(f"Log_{name}.pkl", "wb"))
    pickle.dump(best_rf_cls, open(f"Rf_{name}.pkl", "wb"))
    pickle.dump(best_nn_cls, open(f"Nn_{name}.pkl", "wb"))
