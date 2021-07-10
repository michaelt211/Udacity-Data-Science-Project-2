# Udacity-Data-Science-Project-2


 Disaster Response Pipeline Project
 
This is a 3 part project. The 1st part of this project extracts transforms and loads message data into a database. The 2nd part of the project classifies the message data into categories using a machine learning pipeline and the 3rd part of the project visualizes the message data using a flask app. 

- run.py --> runs the flask web app 
- process_data.py --> runs the ETL pipeline 
- train_classifier2.py --> runs the machine learning pipeline 
- DisasterResponse10.db --> database with clean processed data 
- classifier.pkl --> Model file 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse10.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier2.py data/DisasterResponse10.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/  or https://view6914b2f4-3001.udacity-student-workspaces.com/ 


References: 

- Stack Overflow 
- Python and library documentation 
