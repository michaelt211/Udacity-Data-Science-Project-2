import sys




#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline Preparation
# Follow the instructions below to help you create your ETL pipeline.
# ### 1. Import libraries and load datasets.
# - Import Python libraries
# - Load `messages.csv` into a dataframe and inspect the first few lines.
# - Load `categories.csv` into a dataframe and inspect the first few lines.

# In[1]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split







def load_data(messages_filepath, categories_filepath):
    
    """
    Load Data from csv files merge files into single dataframe
    Args: 
        messages_filepath, categories_filepath
    Returns 
       Merged DataFrame
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    #messages = pd.read_csv('disaster_messages.csv')
    messages.head()


    # In[3]:


    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #categories = pd.read_csv('disaster_categories.csv')
    categories.head()

    df = categories.merge(messages, left_on = 'id', right_on = 'id' )
    df.head()
    return df

def clean_data(df):
    """
    clean data by splitting columns and getting rid of duplicates and converting data types
    Args: 
       data frame
    Returns 
      clean data frame
    """   
    categories_split = df.categories.str.split(";",expand=True)
    categories_split = categories_split.applymap(str)
    categories_split_df_list=categories_split
    categories_split.columns = categories_split_df_list.iloc[0, :] 
    categories_split_df_list.head()
    row = categories_split_df_list.iloc[1,:]
    category_colnames = row.str.split(pat = '-', expand = True)[0].tolist()
    print(category_colnames)
    categories_split_df_list.columns = category_colnames
    categories_split_df_list.head()
    categories_split_df_list




    categories_split_df_list_1 = categories_split_df_list.astype(str)

    for column in categories_split_df_list_1 :
    #set each value to be the last character of the string
        categories_split_df_list_1[column] = categories_split_df_list_1[column].str.split("-", n = 1, expand = True)[1]
    # convert column from string to numeric
    #categories[column] = categories[column]
#categories
    print(categories_split_df_list_1 )
# ### 5. Replace `categories` column in `df` with new category columns.
# - Drop the categories column from the df dataframe since it is no longer needed.
# - Concatenate df and categories data frames.




# drop the original categories column from `df`
    del df['categories']
    print(df)
    print(categories_split_df_list_1)



# concatenate the original dataframe with the new `categories` dataframe
    df1 = pd.concat([categories_split_df_list_1,df], axis = 1)
    df1 



    df1.count()[0] - df1.drop_duplicates().count()[0]
# of entries entries for each of the unique rows
    dups = df1.pivot_table(index = list(df1.columns), aggfunc = 'size')

# In[15]:
#drop duplicates 
    
    df2 = df1.drop_duplicates()
    df2 = df2[df2['related'] != '2']
# In[16]:
# check number of duplicates
    df2.count()[0] - df2.drop_duplicates().count()[0]

  
    return df2

def save_data(df, database_filename):
    
    
    
    """
    
    saves dataframe to database file
    Args: 
       data frame and database_filename
    Returns 
      none
    
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response_table', engine, index=False)


def main():
    #python process_data.py disaster_messages.csv disaster_categories.csv sqlite:///DisasterResponse.db
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
    
    
    