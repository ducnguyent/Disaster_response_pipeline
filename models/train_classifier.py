import sys
import pickle
import re
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


nltk.download(['punkt', 'wordnet', 'stopwords'])
stop_words = set(nltk.corpus.stopwords.words('english'))


def load_data(database_filepath):
    """Load data from database file

    Args:
        database_filepath (str): database filepath

    Returns:
        tuple: Include data X, Y and list of category_name
    """
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("messages_categories", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = df.columns[4:]

    return X, Y, category_names


def tokenize(text):
    """Function to tokenize text.

    Args:
        text (str): input text string

    Returns:
        list: text clean token
    """
    # Lower case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Tokenize
    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)

    # Remove stop words
    clean_tokens = [word for word in clean_tokens if word not in stop_words]
    return clean_tokens


def build_model():
    """Builds model as pipeline
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 50],
        'clf__estimator__min_samples_leaf': [2],
        'clf__estimator__min_samples_split': [2],

    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model perfomace

    Args:
        model (Classifier): trained model
        X_test (pd.Series): Input test dataset
        Y_test (pd.Series): Ouput label test dataset
        category_names (list): List of category names
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """Save trained model to pickle file

    Args:
        model (Classifier): trained model
        model_filepath (str): path to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
