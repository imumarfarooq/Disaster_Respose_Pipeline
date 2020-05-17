import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import re
import nltk
nltk.download(['punkt' , 'wordnet'])


from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")


def load_data(database_filepath):
    
    df_file_name = 'sqlite:///' + database_filepath
    engine = create_engine(df_file_name)
    df = pd.read_sql_table("Disater_messages" , con=engine)
    
    X = df["message"]
    Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
    category_names = Y.columns
    
    return X,Y,category_names
    


def tokenize(text):
    
    tokens = word_tokenize(text)
    lem = WordNetLemmatizer()
    
    token_clears = []
    for key_tok in tokens:
        token_cl = lem.lemmatize(key_tok).lower().strip()
        token_clears.append(token_cl)
    return token_clears
    


def build_model():
    
    RandomForest_pipeline = Pipeline([
    ("vect" , CountVectorizer(tokenizer=tokenize)),
    ("tfidf" , TfidfTransformer()),
    ("clf" , MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {'clf__estimator__max_depth': [10, 50, None],
                  'clf__estimator__min_samples_leaf':[2, 5, 10]}
    
    cv = GridSearchCV(RandomForest_pipeline, parameters)
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    results = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])
    num = 0
    for category in Y_test.columns:
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[category], y_pred[:,num], average='weighted')
        results.set_value(num+1, 'Category', category)
        results.set_value(num+1, 'f_score', f_score)
        results.set_value(num+1, 'precision', precision)
        results.set_value(num+1, 'recall', recall)
        num += 1
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))

    return results
    
    
    
    
    
    


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath , 'wb'))
    


def main():
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
