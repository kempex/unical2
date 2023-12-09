import dash
from dash import Dash, html, dcc, dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import neighbors, naive_bayes
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from random import randint
import random
import pickle
import time
#from views import firstView

myvar = "My Dashboard",


pipe = Pipeline([("classifier", RandomForestClassifier())])

classifiers_dict = [
                    {"classifier": [LogisticRegression()],
                                   "classifier__penalty": ["l2"],
                                   "classifier__C": [100, 10, 1.0, 0.1, 0.01],
                                   "classifier__solver": ['newton-cg', 'lbfgs', 'liblinear']
                    },
                    {"classifier": [KNeighborsClassifier()],
                                   "classifier__n_neighbors": range(1, 21, 2),
                                   "classifier__weights": ['uniform', 'distance'],
                                   "classifier__metric": ['euclidean', 'manhattan', 'minkowski']
                    },
                    {"classifier": [GaussianNB()],
                                   "classifier__var_smoothing": [0.065],
                                   "classifier__priors": [None]
                    },
                    {"classifier": [DecisionTreeClassifier()],
                                   "classifier__criterion": ['gini', 'entropy'],
                                   "classifier__max_depth": [2,4,6,8,10,12]                    
                    },
                    {"classifier": [RandomForestClassifier()],
                                   "classifier__n_estimators": [10, 100, 1000],
                                   "classifier__max_features": [1, 2, 3]
                    },
                    {"classifier": [GradientBoostingClassifier()],
                                   "classifier__n_estimators": [10, 100, 1000],
                                   "classifier__max_depth": [3, 7, 9],
                                   "classifier__learning_rate": [0.001, 0.01, 0.1],
                                   "classifier__subsample": [0.5, 0.7, 1.0]                    
                    }
                ]




'''[
                    {"classifier": [LogisticRegression()],
                                   "classifier__penalty": ["none", "l2"],
                                   "classifier__C": np.logspace(0, 2)
                    },
                    {"classifier": [RandomForestClassifier()],
                                   "classifier__n_estimators": [10, 50],
                                   "classifier__max_features": [1, 2]
                    }
                ]'''


models_selection = {'Logistic Regression': 0,
          'KNN': 1,
          'Naive Bayes': 2,
          'Decision Tree': 3,
          'Random Forest': 4,
          'Gradient Boosting': 5,
          'Find the Best Algorithm': 6}


models = {'Logistic-Regression': LogisticRegression,
          'k-NN-classifier': neighbors.KNeighborsClassifier,
          'Naive Bayes': naive_bayes.GaussianNB,
          'Find the Best Algorithm': 'find' }

all_parameters = {
    'Naive Bayes': [u'var_smoothing', 'priors'],
    'Logistic-Regression': [u'penalty', 'C'],
    'k-NN-classifier': [u'n_neighbors', 'p']         
}

all_paramValues = {
    'Naive Bayes': [0.065, 'None'],
    'Logistic-Regression': ['l2', 0.01],
    'k-NN-classifier': [12, 0.01],       
}

Prediction_Class = {
    '1': 'None Diabetes',
    '0': 'Diabetes',  
}

# Load dataset
dataset = pd.read_csv("C:/Users/HP/Desktop/School/Class - IOT/FinalProject/diabetes_diagnostic.csv")
displayDataset = dataset.iloc[:, [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15]].head()


################### Preprocessing the data
# Converting 'Diebetes' and 'Gender' columns  to numeric
diabetes = dataset.iloc[:,15]
gender = dataset.iloc[:,6]
le = preprocessing.LabelEncoder()
diabetes=le.fit_transform(diabetes)
gender=le.fit_transform(gender)
dataset['EncodedDiabetes'] = diabetes
dataset['EncodedGender'] = gender

# cholesterol, glucose, hdl_chol, age, height, weight, systolic_bp, waist, hip
X = dataset.iloc[:, [1, 2, 3, 5, 7, 8, 10, 12, 13]]                 

# target variable
target_diabetes = dataset.iloc[:,16]

# pick a sample row for prediction
sampleData = X.sample(n = 1)
#sampleData = X.iloc[[64]]



# SPlitting values for training and testing
def getData(testSize=80, trainSize=250):
    x_train, x_test, y_train, y_test = train_test_split(X,target_diabetes, test_size=testSize, train_size=trainSize)

    return x_train, x_test, y_train, y_test


# Set the ploty for confusion matrix
df = px.data.tips()

app = dash.Dash(__name__)
#application = app.server



#################################### Set the sample values
@app.callback( 
    Output('sample-table', component_property='children'),               
    Input('sample_type', 'value'))

def predict(value):
    # Find a random record from X df
    sampleData = X.sample(n = 1)
    # Empty the df if new record is selected
    if value == 'Enter Record':
        for col in sampleData.columns:
            sampleData[col].values[:] = 0

    sampleDiv = generate_sample_table(sampleData)
    return sampleDiv



#################################### Predit the sample data
@app.callback( 
    Output('sample-result-label', component_property='children'),  
    Output('sample-probability-label', component_property='children'),                 
    Input('predict-button', 'n_clicks'),
    State('table-editing-simple', 'data'),
    State('table-editing-simple', 'columns'), prevent_initial_call=True)

def predict(n_clicks_predict, rows, columns):
    df = pd.DataFrame(rows, columns=[c['name'] for c in columns])
    sampleX = df.iloc[[0]]
    # load the model from disk
    filename = 'finalized_model.sav'
    model = pickle.load(open(filename, 'rb'))

    y_predicted_Sample = model.predict(sampleX)
    sample_y_predicted = Prediction_Class[str(y_predicted_Sample.flat[0])]
    y_probability_Sample = model.predict_proba(sampleX)
    sample_y_probability = max(round(y_probability_Sample.flat[0], 2), round(y_probability_Sample.flat[1], 2))
    if sample_y_predicted == 'Diabetes':
        sample_result = html.H3('Prediction: ' + sample_y_predicted, style={"margin-left": "0px", "backgroundColor": "red"})
    else:
        sample_result = html.H3('Prediction: ' + sample_y_predicted, style={"margin-left": "0px", "backgroundColor": "lightgreen"})

    return sample_result, sample_y_probability


#################################### Train the model and Display the Graph
@app.callback(
    Output('graph_heatmap_confusion', 'figure'),
    Output('pie-chart', 'figure'),
    Output('accuracy_rate', component_property='children'),
    Output('f1-score', component_property='children'),
    Output('mse-score', component_property='children'),
    Output('selected-model', component_property='children'),
    Output('selected-model-label', component_property='style'),
    Output('submit-button', component_property='children'),
    Output('predict-button', component_property='style'),     
    Input('submit-button', 'n_clicks'),
    State('model-name', 'value'),
    State('my-slider', 'value'), prevent_initial_call=True)

def train_and_display(n_clicks, name, sliderValue):
    
    # Pre-process the dataset
    x_train, x_test, y_train, y_test = getData(390-sliderValue, sliderValue)
    selected_model = ''

    # Find the best classifier
    if name == 'Find the Best Algorithm':
        classifier_obj = classifiers_dict

    # Use selected classifier
    else:
        classifier_obj = classifiers_dict[models_selection[name]]

    # Find the best hyperparameters
    clf = GridSearchCV(pipe, classifier_obj, cv=5, verbose=0, n_jobs = -1)
    # Train the model
    model = clf.fit(x_train, y_train)   
    selected_model = str(model.best_estimator_["classifier"])

    print('final model: ' + str(model))

    # save the model to disk
    filename = 'finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))

    # predict the validation data for evaluating the performance
    y_predicted = model.predict(x_test)

    # Create Confusion Matrix
    cm= confusion_matrix(y_test, y_predicted)
    resultStr = str(cm)
    # Calculate Accuracy Score
    accuracy = "Classification Accuracy: {:.2%}".format(accuracy_score(y_test, y_predicted))
    accuracy_rate = accuracy
    # Calculate F1-Score
    f1Score = "F1 Score: {:.4}".format(f1_score(y_test, y_predicted))
    # Calculate Mean Squared Error (MSE)
    mseScore = "Mean Squared Error: {:.4}".format(mean_squared_error(y_test, y_predicted))

    # Confusion Matrix and the percentage chart
    fig_pie = px.pie(df, values=[cm[0][0], cm[0][1], cm[1][0], cm[1][1]], names=['True Negative', 'False Positive', 'False Negative', 'True Positive'])
    fig = px.imshow(cm, labels=dict(x="Predicted Values", y="True Values"), aspect="auto")

    label_property = {'display': 'block', 'font-weight': 'bold'}
    button_value = 'Train the model'
    buttonStyle = {'pointer-events': 'initial', 'opacity': '1'}

    return fig, fig_pie, accuracy_rate, f1Score, mseScore, selected_model, label_property, button_value, buttonStyle


# Create table for the sample data
def generate_sample_table(dataframe, max_rows=10):

    params_list =  dataframe.columns.tolist()
    #params_list_values =  dataframe.iloc[0].tolist()
    #params_list_values_str = [str(x) for x in params_list_values]
    #print("params list: ", params_list_values_str)
    sampleData_dict = dataframe.to_dict('records')
    #print("dict: ", sampleData_dict)

    return dash_table.DataTable(
        id='table-editing-simple',
        columns=(
            [{'id': p, 'name': p} for p in params_list]
        ),
        data=sampleData_dict,
        editable=True
    )



# Create table for the dataset
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col, style = {'border': '1px solid', 'padding': "1px", 'backgroundColor':'lightblue'}) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col], style = {'border': '1px solid', 'padding': "1px", 'textAlign':'center'}) for col in dataframe.columns
            ], style = {'border': '1px solid'}) for i in range(min(len(dataframe), max_rows))
        ])
    ], style = {'border': '1px solid'})


# Create the correlation table
def generateCorrelationTable():
    f, axx = plt.subplots(figsize=(10,10))
    fig = px.imshow(dataset.iloc[:,1:].corr(), labels=dict(x="Predicted Values", y="True Values"), aspect="auto")   
    graphElement = dcc.Graph(id="graph_correlation", children=[fig])
    return graphElement


# Show the value of slider
@app.callback(
    Output('slider-output-container', 'children'),
    Input('my-slider', 'value'))
def update_output(value):
    return 'You have selected "{}" records for training data'.format(value)

'''
@app.callback(
    Output('predict-button', component_property='style'),       
    Input('submit-button', 'n_clicks'),  prevent_initial_call=True)

def updateButton(n_clicks):
    buttonStyle = {'pointer-events': 'initial', 'opacity': '0.3'}
    return buttonStyle
'''


################################# Layout ########################################

app.layout = html.Div([

    ########################################## Heading Section ########################################## 
    html.H1('Predict Diabetes App - Final Project - Meghdad Sanii'),

    ##########################################  Top Div Section ########################################## 
    html.Div([

        ############## First Div Section ##############
        html.Div([

            html.H2("Select Model:"),
            dcc.Dropdown(id='model-name',options=[{'label': x, 'value': x} for x in models_selection], value='Logistic Regression', clearable=False),
            html.Br(),
            html.Label(id='selected-model-label', children='Selected Algorithm & Hyperparameters (based on GridSearch):', style={'font-weight': 'bold', 'display': 'none'}),
            html.Br(),
            html.Label(id='selected-model', children='', className='output-example-loading')

        ]), 
        # End of First Div section

        html.Div([

            html.H2("Training data split:"),
            dcc.Slider(50, 300, 20, value=200, id='my-slider'),
            html.Div(id='slider-output-container')

        ]), 
        # End of First Div section

        html.Div([
            html.Br(),
            html.Br(),
            dcc.Loading(id="loading-Element", className='loadingCircle2', children=html.Button(children='Train the model', id='submit-button', className='trainButton'), type="default")
        
        ]), 
        # End of First Div section


    ], className="topSection", style={'display': 'flex', 'flex-direction': 'row', 'padding': 0, 'border-style': 'solid'}), 
    # End of Top Section

    ##########################################  Sample Div Section ########################################## 
    html.H2('Sample Record: '),
    dcc.RadioItems(id='sample_type', options=['Sample Record', 'Enter Record'], value='Sample Record', inline=True),
    html.Div([

        # Sample data
        html.Div(id='sample-table', children=[dash_table.DataTable(id='table-editing-simple')], style = {'flex': '30%'}),
        html.Button('Predict', id='predict-button', className='predictButton', style = {'pointer-events': 'none', 'opacity': '0.4'}),   
        html.Div(children=[
            html.Div(id='sample-result-label', children='', style={"margin-left": "60px"})], 
            style={'flex': '10%', 'backgroundColor': 'lightgray', 'display': 'flex', 'margin-left': 50, 'padding': 10, 'border-style': 'outset'}),
        html.Div(children=[
            html.H3('Probability: '),
            html.H3(id='sample-probability-label', children='',  style={"margin-left": "20px"})], style={'flex': '10%', 'backgroundColor': 'lightgray', 'display': 'flex', 'margin-left': 20, 'padding': 10, 'padding-left': 100, 'border-style': 'outset' 
            }),

    ], style={'display': 'flex', 'flex-direction': 'row', 'padding': 10, 'border-style': 'groove'}), 
    # End of Sample Div Section



    ##########################################  Middle Div Section ########################################## 
    html.Div([

        ############## First Div Section ##############
        dcc.Loading(id='loading_performance', children=html.Div([

            html.H2("Model Performance"),
            html.H3(id='accuracy_rate', children='', style={'backgroundColor': 'lightgray', 'display': 'flex', 'padding': 10, 'border-style': 'outset' }),
            html.H3(id='f1-score', children='', style={'backgroundColor': 'lightgray', 'display': 'flex', 'padding': 10, 'border-style': 'outset' }),
            html.H3(id='mse-score', children='', style={'backgroundColor': 'lightgray', 'display': 'flex', 'padding': 10, 'border-style': 'outset' }),

        ]), type="circle"),

        # End of First Div section
 
        ############## Second Div Section ##############
        dcc.Loading(id='confusion_matrix', children=html.Div([

            html.H2("Confusion Matrix"),
            dcc.Graph(id="graph_heatmap_confusion"),

        ]), type="circle"),
        # End of Second Div section
 
        ############## Third Div Section ##############
        dcc.Loading(id='confusion_chart', children=html.Div([

            html.H2(children='Confusion Matrix - Percentage'),
            dcc.Graph(id="pie-chart"),

        ]), type="circle"),
        # End of Third Div Section     

    ], style={'display': 'flex', 'flex-direction': 'row', 'padding': 10, 'border-style': 'none'}, className='model-performance-outerDiv'), 
    # End of Middle Section



    ##########################################  Bottom Div Section ########################################## 
    html.Div([

        ############## First Div Section ##############
        html.Div([

            html.H2("Original Dataset"),
            generate_table(displayDataset),

        ], style={'width': '100%', 'display': 'flex', 'flex-direction': 'column','padding': 10, 'border-style': 'groove'}), 
        # End of First Div section
 
    ], style={'display': 'flex', 'flex-direction': 'row', 'padding': 10, 'border-style': 'none'}), 
    # End of Bottom Section

], style={'border':'1px'})



if __name__ == "__main__":
    app.run_server(debug=True)
