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


# In[2]:


# load messages dataset
messages = pd.read_csv('disaster_messages.csv')
messages.head()


# In[3]:


# load categories dataset
categories = pd.read_csv('disaster_categories.csv')
categories.head()


# ### 2. Merge datasets.
# - Merge the messages and categories datasets using the common id
# - Assign this combined dataset to `df`, which will be cleaned in the following steps

# In[4]:


# merge datasets
df = categories.merge(messages, left_on = 'id', right_on = 'id' )
df.head()


# ### 3. Split `categories` into separate category columns.
# - Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
# - Use the first row of categories dataframe to create column names for the categories data.
# - Rename columns of `categories` with new column names.

# In[5]:


# create a dataframe of the 36 individual category columns
categories_split = df.categories.str.split(";",expand=True)
categories_split = categories_split.applymap(str)
categories_split_df_list=categories_split
categories_split.columns = categories_split_df_list.iloc[0, :] 

#results_analysis_df_js_unique = np.unique(categories_split_df_list[[0:].values)
categories_split_df_list.head()


# In[6]:


# select the first row of the categories dataframe
row = categories_split_df_list.iloc[1,:]

#row.str.split(pat = '-', expand = True)[0].tolist()

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
#row.lambda()
category_colnames = row.str.split(pat = '-', expand = True)[0].tolist()
print(category_colnames)


# In[7]:


# rename the columns of `categories`
categories_split_df_list.columns = category_colnames
categories_split_df_list.head()
categories_split_df_list


# ### 4. Convert category values to just numbers 0 or 1.
# - Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
# - You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.

# In[8]:


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

# In[9]:


# drop the original categories column from `df`
del df['categories']


# In[10]:


print(df)


# In[11]:


print(categories_split_df_list_1)


# In[12]:


# concatenate the original dataframe with the new `categories` dataframe
df1 = pd.concat([categories_split_df_list_1,df], axis = 1)
df1 


# ### 6. Remove duplicates.
# - Check how many duplicates are in this dataset.
# - Drop the duplicates.
# - Confirm duplicates were removed.

# In[13]:


# check number of duplicates
#duplicate = df[df.duplicated()] # returns duplicated values 
#dups = df1.pivot_table(index = list(df1.columns), aggfunc = 'size')
#type(dups)


# In[14]:


# of duplicates
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



# ### 7. Save the clean dataset into an sqlite database.
# You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.

# In[17]:


engine = create_engine('sqlite:///DisasterResponse.db')
df2.to_sql('message_and_categories_ds10', engine, index=False)


# In[ ]:





# ### 8. Use this notebook to complete `etl_pipeline.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.

# In[ ]:





# In[ ]:






def load_data(messages_filepath, categories_filepath):
    pass


def clean_data(df):
    pass


def save_data(df, database_filename):
    pass  


def main():
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
    
    
    