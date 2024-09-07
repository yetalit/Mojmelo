import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def get_data():
    # Load the SMS Spam Collection Dataset
    sms_data = pd.read_csv("spam.csv", encoding='latin-1') # url: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
    sms_data = sms_data[['v1', 'v2']]

    return [CountVectorizer().fit_transform(sms_data['v2']).toarray(), sms_data['v1']]
