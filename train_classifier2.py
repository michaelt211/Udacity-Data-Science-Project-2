import sys
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
import warnings
#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y
def load_data(database_filepath):
    df = pd.read_sql_table('message_and_categories_ds10', con = 'sqlite:///' + database_filepath)
    X = df.loc[:,["message"]]
    X = X.squeeze()
    Y = df.iloc[:,list(range(36))] 
    category_names = list(Y.columns.values)
    return X, Y, category_names


def tokenize(text):
    """ Normalize text string, tokenize text string and remove stop words from text string
    Args: 
        Text string with message
    Returns 
        Normalized text string with word tokens 

    """
    
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize words
    tokens = word_tokenize(text)
    
    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return stemmed


def build_model():
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {
        'vect__smooth_idf': [True,False],
    }
    # create grid search object

    cv = GridSearchCV(pipeline, param_grid=parameters,cv =4,verbose = 10)
    return cv

def get_eval_metrics(actual, predicted, col_names):
    """Calculate evaluation metrics for ML model
    
    Args:
    actual: array. Array containing actual labels.
    predicted: array. Array containing predicted labels.
    col_names: list of strings. List containing names for each of the predicted fields.
       
    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall 
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []
    
    # Calculate evaluation metrics for each set of labels
    for i in range(len(col_names)):
        #print(col_names[i])
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i])
        recall = recall_score(actual[:, i], predicted[:, i])
        f1 = f1_score(actual[:, i], predicted[:, i])
        
        metrics.append([accuracy, precision, recall, f1])
    
    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data = metrics, index = col_names, columns = ['Accuracy', 'Precision', 'Recall', 'F1'])
      
    return metrics_df  

def evaluate_model(model, X_test, Y_test, category_names):
    y_test_pred = model.predict(X_test).astype(int)
    #y_test_pred = y_test_pred.astype(int)
    eval_metrics =get_eval_metrics(np.array(Y_test).astype(int), y_test_pred, category_names)
    print(eval_metrics)

def save_model(model, model_filepath):
    """Pickle fitted model
    
    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    
    Returns:
    None
    """
    # Pickle the model
    pickle.dump(model, open(model_filepath, 'wb'))
    #pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))
  

def main():
    #python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    #python train_classifier.py DisasterResponse10.db classifier.pkl
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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