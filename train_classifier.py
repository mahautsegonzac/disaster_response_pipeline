# import libraries
import sys
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    This function extracts the data from a SQL database and splits it into features and labels
    Arguments:
        database_filepath = path to SQL database containing table
    Output:
        X = features (array of strings)
        Y = labels (array of integers)
        category_names = list of categories (strings) to which a message can belong (or label names)
    '''
    # load data from database
    print('testing engine')
    engine = create_engine(database_filepath)
    print(engine.table_names())

    df = pd.read_sql("SELECT * FROM messages_table", engine)
    X = df['message'].values
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    '''
    This function will be used to clean the text making up the features in order to make it apt to feed a classifier
    Arguments:
        text = string
    Output:
        message = list of normalized, unpunctuated, lemmatized words, excluding stop words
    '''
    # Normalize string and remove punctuation
    message = word_tokenize(re.sub(r'[^a-zA-Z0-9]', ' ', text.lower()))
    # Remove stop words
    message = [word for word in message if word not in stopwords.words("english")]
    # Lemmatize the words
    message = [WordNetLemmatizer().lemmatize(word) for word in message]
    return message

def build_model():
    '''
    This function builds the model which will be used to classify messages
    Output:
        model = machine learning pipeline with optimized set of parameters
    '''
    # Instantiate pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('model', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Before using GridSearch to fine-tune hyperparameters, check the list of available parameters
    print(pipeline.get_params().keys())
    
    # Run GridSearch on pipeline with chosen parameters and range of values
    parameters = {
#         'model__estimator__n_estimators': [100, 500, 1000],
        'model__estimator__max_depth': [5, 15, 30],
        'model__estimator__min_samples_split':[2, 10, 15],
#         'model__estimator__min_samples_leaf':[1, 2, 5, 10],
        'model__estimator__bootstrap': [True, False]
    }
    
    # Build model with best combination of parameters
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model

def evaluate_model(model, X_test, Y_test, category_names): 
    '''
    This function uses model to make predictions and assesses accuracy of these predictions
    Arguments:
        model =  machine learning pipeline already trained
        X_test = test features
        Y_test = test labels
        category_names = list of label names
    '''
    # Predict values on test data using pipeline
    y_pred = model.predict(X_test)
    # Compute recall, precision and f1 score
    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    '''
    This function saves the model as a pickle file
    Arguments:
        model = trained model
        model_filepath = name under which the model will be saved
    '''
    # Write a pickled representation of the tuned model
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)

def main():
    sys.argv = ['train_classifier.py', 'sqlite:///DisasterResponse.db', 'tuned_classifier.pkl']
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        # Perform train test split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()