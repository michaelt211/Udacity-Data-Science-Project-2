import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Histogram
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sys






app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('message_and_categories_ds10', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_cat = df.iloc[:,:37]
    df_cat = df_cat.apply(pd.to_numeric).sum().to_frame()
    df_cat.reset_index(inplace = True)
    x_cat = df_cat.iloc[:36,0]
    y_cat_count = df_cat.iloc[:36,1]
    
    df['message_len'] = df['message'].apply(len)
    y_length_count = df['message_len']
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    # Bar Chart by Genere and Counts 
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'categoryorder':'total descending'
                }
            }
        }, 
        
        
        
        
        
        
        
        
        
        
        
        
        # Bar Chart by Category and Counts 
        
        
       {
            'data': [
                Bar(
                    x= x_cat,
                    y=y_cat_count
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category", 
                    'categoryorder':'total descending'
                    
                   
                }
            }
        } , 
        
        
        
        
        
        
        
        
        # Bar Chart by Category and Counts 
        
        {
            'data': [
                Histogram(
                   x =  y_length_count, xbins=dict(start=np.min(y_length_count), end =np.percentile(y_length_count, 99))
              
               
                )
            ],

            'layout': {
                'title': 'Distribution of Lengths',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Length"
                }
            }
        } 
        
        
        
        
        
        

        
        
        
        
       
        
        
        
        
        
        
    ]
    
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0].astype(int)
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()