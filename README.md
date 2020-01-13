# Disaster Response Pipeline

This project aims at classifying messages received during disaster events by category, so that they can be sent to the appropriate disaster relief agency.
For this purpose, I have built a machine learning pipeline using a classifier performing speech recognition.
This project also includes a web app where an emergency worker can input a new message and get classification results in several categories. 
The datasets I used contain real messages sent during disaster events and have been provided by the company [Figure Eight](https://www.figure-eight.com/).


### How to 

- Run the Python scripts : 
    - Run the following commands in the project's root directory to set up the database and model : `python process_data.py`, `python train_classifier.py`.

- Run the web app : 
    - Run the following command in the root directory to run the web app : `python run.py`
    - Go to http://0.0.0.0:3001/


### Libraries

The data preparation and modeling have been executed in Python 3.7.5.
The following libraries need to be installed to clean the data, build and train the model, and create the web app:

    sys
    pickle
    numpy
    pandas
    sqlalchemy
    sklearn
    re
    nltk
    flask
    plotly
    json


### Files in Repository

    process_data.py - contains an ETL, returning a clean dataset in a SQL database ('DisasterResponse.db')
    train_classifier.py - contains a machine learning pipeline, preparing the text data in order to make it appropriate for modelisation and feeding a random forest classifier with the prepared data. This file returns the model saved as a pickle file.
    run.py - creates the flask web app (this script has been entirely provided by Udacity)

    disaster_categories.csv - csv file containing data related to the category assigned to each message
    disaster_messages.csv - csv file containing data related to the content of the messages sent

    go.html - template used when building the web app
    master.html - template used when building the web app
