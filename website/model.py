import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle
from sklearn.linear_model import LogisticRegression


#Took out germanAppliance2, germanOther, germanRetraining2, germanEducation2
file_list = ['germanBusiness.csv', 'germanCarNew.csv', 'germanCarUsed.csv',
            'germanFurniture.csv', 'germanRadio_TV.csv', 'germanRepairs.csv']

# Initialize an empty dictionary to store DataFrames
dataframes = {}

# Loop through the files and read them into DataFrames
for file in file_list:
    df = pd.read_csv(f"../datasets/cleanData/{file}")
    dataframes[file[:-4]] = df

for name, df in dataframes.items():
    train, test = train_test_split(df, test_size=0.20, random_state=12)
    # The remaining
    X_train, y_train = train.drop(columns=['Class']) ,train['Class']
    X_test, y_test = test.drop(columns=['Class']), test['Class']

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test= sc.transform(X_test)

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
    best_cls = grid_search.best_estimator_

    # Make pickle file of our model
    pickle.dump(best_cls, open(f"{name}_model.pkl", "wb"))