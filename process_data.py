# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the two datasets needed for modeling
    Arguments:
        messages_filepath = path to disaster_messages csv file
        categories_filepath = path to disaster_categories csv file
    Output:
        df = dataframe made of these two datasets
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on=['id'])
    return df

def clean_data(df):
    '''
    This function performs a series of cleaning actions to make the data ready for modeling
    Arguments:
        df = dataframe to clean
    Output:
        df = clean dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,:]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x : x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-')[1][1]
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # check number of duplicates
    print('{} duplicated rows'. format(df.duplicated().sum()))
    # drop duplicates
    df.drop_duplicates(inplace=True)
    # check number of duplicates
    print('{} duplicated rows after deletion'. format(df.duplicated().sum()))
    return df

def save_data(df, database_filename):
    '''
    This function saves the dataframe in a SQL database
    Arguments:
        df = dataframe to be loaded to the database
        database_filename = path to database
    '''
    engine = create_engine(database_filename)
    df.to_sql('messages_table', engine, if_exists='replace', index=False)
    print(engine.table_names())

def main():
    sys.argv = ['process_data.py', 'disaster_messages.csv', 'disaster_categories.csv', 'sqlite:///DisasterResponse.db']
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()