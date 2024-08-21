from sklearn.model_selection import train_test_split
import pandas as pd

def get_data():
    data = pd.read_csv("BostonHousing.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    return [X_train, X_test, y_train, y_test]
