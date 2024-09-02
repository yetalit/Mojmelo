from sklearn.datasets import make_classification
import numpy as np

def get_data():
    # Generate a dataset with continuous features
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.3, 0.4, 0.3],
        random_state=42
    )
    # Convert continuous features to discrete features
    # Here, we use binning to simulate discrete feature values
    X_binarized = np.digitize(X, bins=np.linspace(X.min(), X.max(), 5))

    return [X_binarized, y]
