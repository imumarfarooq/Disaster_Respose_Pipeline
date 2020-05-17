# Disaster_Respose_Pipeline
## Objective
In this project, I'm applying Data Engineering  techniques to analyze the data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> to build a classification model for an API that classify the Disaster message into one of the given 36 categories and contact the particular category company to help these peoples.

In **_data_, Directory** contains the dataset of real Disaster messages amd categories. You will also get the update data from <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a> website. 
I use these datasets to perform ETL operations to clean the data that can easily feed into the **Machine Learning** model and get better results. In Data Engineering, the main task is to clean the data and make a pipeline to use multiple algorithms at a time and get better accuracy. 

To make it useable for not a technical person, I'll make a user-friendly web-app and help it, users, to easily type their queries and get the response from the appropriate relief agency.

## Project Components
### 1 ETL Pipeline
- Load the dataset ('Disaster_messages.csv and Disaster_categories.csv')
- Merge the dataset
- Perform Data wrangling techniques to clean tha data
- Store the data into **SQlite Database**

### 2 ML Pipeline
 - Load data from **SQlite Database**
 - Split the data into testing and training set
 - Build a pipeline of text processing (NLP) and **Machine Learning Model**
 - Train and tune the model using **GridSearchCV**
 - Predict the output on test data
 - Export the final model into **modelname.pkl** format
 
 ### 3 Flask Web-App
 - Load the Database and model
 - run the command python **app/run.py**
 - webpage is open in a given address **http://0.0.0.0:3001/**
 
 
 
