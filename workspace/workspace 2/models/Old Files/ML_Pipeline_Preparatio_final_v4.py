#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings
#!pip install numpy --upgrade
warnings.simplefilter('ignore')


# In[2]:


import sqlite3 
con = sqlite3.connect("DisasterResponse10.db")
cursor = con.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())


# In[3]:


# load data from database
#df = cursor.execute("SELECT * FROM message_and_categories_ds4 ")
#engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql_table('message_and_categories_ds10', con = 'sqlite:///DisasterResponse10.db')
df.to_csv('ML_Data.csv')
df.head()


# In[4]:


X = df.loc[:,["message"]]
X = X.squeeze()
Y = df.iloc[:,list(range(36))]


# In[5]:


for column in Y.columns:
    print(column, ': ',  Y[column].unique())
Y = Y.apply(pd.to_numeric)


# In[6]:


Y.dtypes


# In[7]:


X


# In[8]:


Y


# ### 2. Write a tokenization function to process your text data

# def tokenize(text):
#     # Convert text to lowercase and remove punctuation
#     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#     clean_tokens = []
#  # Stem word tokens and remove stop words
#     stemmer = PorterStemmer()
#     stop_words = stopwords.words("english")
#     
#     stemmed = [stemmer.stem(word) for word in tokens if word not in stop_words]
#     
#     return stemmed
# 
# 
# 

# In[9]:


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


# def tokenize(text):
#     """
#     Function: tokenize the text
#     Args:  source string
#     Return:
#     clean_tokens(str list): clean string list
#     
#     """
#     #normalize text
#     text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
#     
#     #token messages
#     words = word_tokenize(text)
#     tokens = [w for w in words if w not in stopwords.words("english")]
#     
#     #sterm and lemmatizer
#     lemmatizer = WordNetLemmatizer()
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).strip()
#         clean_tokens.append(clean_tok)
# 
#     return clean_tokens

# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[10]:


from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(MultinomialNB()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[11]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle=True, random_state=42 )


# In[12]:


x_train


# In[13]:


np.random.seed(17)
pipeline.fit(x_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[14]:


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


# In[15]:


# Calculate evaluation metrics for training set
y_train_pred = pipeline.predict(x_train)
y_train_pred =y_train_pred.astype(int)
col_names = list(Y.columns.values)


# In[16]:


y_test_nb_pred = pipeline.predict(x_test)
y_test_nb_pred =y_test_nb_pred.astype(int)

nb_result_test =get_eval_metrics(np.array(y_test).astype(int), y_test_nb_pred.astype(int), col_names)
nb_result_test


# In[17]:


# Get summary stats for tuned model
nb_result_test.describe()


# In[18]:


Y.sum()/len(Y)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[19]:


def performance_metric(y_true, y_pred):
    """Calculate median F1 score for all of the output classifiers
    
    Args:
    y_true: array. Array containing actual labels.
    y_pred: array. Array containing predicted labels.
        
    Returns:
    score: float. Median F1 score for all of the output classifiers
    """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)
        
    score = np.median(f1_list)
    return score


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[20]:


for column in y_train.columns:
    print(column, ': ',  y_train[column].unique())


# Multinomial Naive Based
# ==========

# In[21]:


from sklearn.naive_bayes import MultinomialNB

mnb_pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
])

mnb_parameters = {'vect__min_df': [1, 5],
              'tfidf__use_idf':[True, False],
              'clf__estimator__alpha':[0, 1]}

mnb_scorer = make_scorer(accuracy_score)

mnb_cv = GridSearchCV(mnb_pipeline, param_grid = mnb_parameters, scoring = mnb_scorer,n_jobs = 8, verbose = 10)

# Find best parameters
np.random.seed(81)
mnb_model = mnb_cv.fit(x_train, y_train)


# In[22]:


mnb_pipeline.get_params()


# In[23]:


# Parameters for best mean test score
mnb_model.best_params_


# In[24]:


y_test_mnb_pred = mnb_cv.predict(x_test)
y_test_mnb_pred =y_test_mnb_pred.astype(int)


# In[25]:


mnb_result_test =get_eval_metrics(np.array(y_test).astype(int), y_test_mnb_pred.astype(int), col_names)
mnb_result_test


# In[26]:


# Get summary stats for tuned model
mnb_result_test.describe()


# Adaboost based Model
# =======

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
pipeline_ab = Pipeline([
    ('vect',TfidfVectorizer(tokenizer=tokenize)),
    ('clf',  MultiOutputClassifier(AdaBoostClassifier()))
])


# In[28]:


pipeline_ab.get_params()


# In[29]:


parameters_ab = {
    'vect__smooth_idf': [True,False],
}
# create grid search object
cv_ab = GridSearchCV(pipeline_ab, param_grid=parameters_ab,n_jobs=8,cv =4,verbose = 10)
cv_ab.fit(x_train, y_train)


# In[30]:


y_test_ab_pred = cv_ab.predict(x_test)
y_test_ab_pred =y_test_ab_pred.astype(int)


# In[31]:


ab_result_test =get_eval_metrics(np.array(y_test).astype(int), y_test_ab_pred.astype(int), col_names)
ab_result_test


# In[32]:


# Get summary stats for tuned model
ab_result_test.describe()


# ### 9. Export your model as a pickle file

# In[34]:


# Pickle best model
pickle.dump(ab_result_test, open('disaster_model.pkl', 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:





# In[ ]:





# In[ ]:




