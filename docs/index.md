<a id="readme-top"></a>

[![Issues][issues-shield]][issues-url]
[![BSD-3-Clause License][license-shield]][license-url]
![CodeQL](https://github.com/yetalit/Mojmelo/workflows/CodeQL/badge.svg)

<br />
<div align="center">
  <a href="https://github.com/yetalit/mojmelo">
    <img src="./images/logo-min.jpg" alt="Logo" width="256" height="256">
  </a>
  <h3 align="center">Mojmelo</h3>
  <p align="center">
    <a href="https://github.com/yetalit/mojmelo/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/yetalit/mojmelo/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

## About The Project

The name `Mojmelo` is derived from the "Mojo Machine Learning" expression. It includes the implementation of Machine Learning algorithms from scratch in pure Mojo.
Here is the list of the algorithms:
* Linear Regression
* Polynomial Regression
* Logistic Regression
* KNN
* KMeans
* HDBSCAN
* DBSCAN
* SVM
* Naive Bayes
    1. GaussianNB
    2. MultinomialNB
* Decision Tree (Regression/Classification)
* Random Forest (Regression/Classification)
* GBDT (Regression/Classification)
* PCA

Preprocessing:
* normalize
* MinMaxScaler
* StandardScaler
* KFold
* GridSearchCV
* LabelEncoder

**Documentation:** https://yetalit.github.io/Mojmelo/docs/_index.html

## Getting Started

If you are not familiar with Mojo projects, you can get started here: https://mojolang.org/docs/manual/get-started/

### Prerequisites

* mojo-compiler 1.0.0b2

Optionally, bellow Python packages can be installed for a better usability and to run tests:
1. Numpy
2. Pandas
3. Scikit-learn
4. Matplotlib

### Installation

There are three ways to install mojmelo: Using Pixi CLI, PyPI CLI and through the source code.

Additionally, completing the setup process (discussed later) is recommended.

#### Pixi CLI

Make sure you have the Modular community channel (https://repo.prefix.dev/modular-community) in your `pixi.toml` file in the channels section, then add mojmelo this way:
```
pixi add mojmelo
```

To start the setup process, run the following command from the `main folder` of your project:
```
bash ./.pixi/envs/default/etc/conda/test-files/mojmelo/0/tests/setup.sh
```

Note: If CPU cache details are available by the OS, benchmarking parts of the setup will be skipped. Otherwise, please try not to run other tasks on your pc during the process for better results.

#### PyPI CLI

Using the command below, the PyPI package containing the source code will be installed from the github repository:
```
pip install "git+https://github.com/yetalit/Mojmelo.git#subdirectory=pypi"
```

Then start the setup process this way:
```
mojmelo-setup
```

Note: If CPU cache details are available by the OS, benchmarking parts of the setup will be skipped. Otherwise, please try not to run other tasks on your pc during the process for better results.

#### Source Code

Mojmelo can also be installed through the source code. This way, you will have the source code in your project.

First, Download `mojmelo` folder and `setup.mojo` file. To start the setup process, run these commands from where `mojmelo` folder and `setup.mojo` file are stored:
```
mojo build setup.mojo -o setup &&
./setup &&
./setup 1 &&
./setup 2 &&
./setup 3 &&
./setup 4 &&
./setup 5 &&
./setup 6 &&
./setup 7 &&
./setup 8 &&
./setup 9 &&
rm -f ./setup
```

Note: If CPU cache details are available by the OS, benchmarking parts of the setup will be skipped. Otherwise, please try not to run other tasks on your pc during the process for better results.

## Usage

Importing models is straightforward:
```python 
from mojmelo.LinearRegression import LinearRegression
```
You may also want to use the utility codes written for this project:
```python 
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import *
```
Here is an example code demonstrating a common training process:
```mojo
from mojmelo.KNN import KNN
from mojmelo.utils.Matrix import Matrix
from mojmelo.preprocessing import train_test_split, GridSearchCV, LabelEncoder
from mojmelo.utils.utils import accuracy_score
from std.python import Python
import std.os as os

def main() raises:
    # Load the Iris dataset from scikit-learn using the Python interoperability API.
    var iris = Python.import_module("sklearn.datasets").load_iris()

    # Create a LabelEncoder instance.
    # This converts class labels into integer values that the model can work with.
    var le = LabelEncoder()

    # Convert the NumPy feature array into a native Matrix.
    var X = Matrix.from_numpy(iris.data)
    # Encode the target labels into integer values.
    var y = le.fit_transform(iris.target)

    # Define the hyperparameter values to test.
    # Here we evaluate KNN with k = 3, 5, and 7.
    var params = Dict[String, List[String]]()
    params["k"] = ["3", "5", "7"]
    # Find the best hyperparameters using grid search.
    # - accuracy_score is the evaluation metric.
    # - cv=4 performs 4-fold cross-validation.
    # - n_jobs=-1 uses all available CPU cores.
    #
    # GridSearchCV returns the best hyperparameters and their score. [0] contains the best parameters.
    var best_params = GridSearchCV[KNN](
        X,
        y,
        params,
        accuracy_score,
        cv=4,
        n_jobs=-1,
    )[0].copy()
    print("Tuned parameters:", best_params)

    # Split the dataset into training and testing sets.
    var X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1234,
    )

    # Create a KNN model using the best hyperparameters found above.
    var knn = KNN(best_params)
    # Train the model using the training data.
    knn.fit(X_train, y_train)
    # Save the trained model to disk.
    knn.save("knn")

    # Load the saved model back from disk.
    knn = KNN.load("knn")
    # Predict the labels for the test samples.
    var y_pred = knn.predict(X_test)
    # Compare the predictions with the expected labels.
    print("KNN classification accuracy:", accuracy_score(y_test, y_pred))

    # Remove the saved model file created by this example.
    os.remove("knn.mjml")
```
More examples are available in [`tests`](https://github.com/yetalit/Mojmelo/blob/main/tests) folder.

## Benchmarks (AMD Zen 4)

[`KMeans`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/kmeans_bench.mojo)

| Model          | Fit Time (s)    | ARI vs sklearn | ARI vs truth |
|----------------|-----------------|----------------|--------------|
| sklearn KMeans | 0.2766 ± 0.0054 |       -        | 0.9390       |
| mojmelo KMeans | 0.2201 ± 0.0089 | 0.8822         | 0.9389       |

[`HDBSCAN`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/hdbs_bench.mojo) (algorithm='boruvka_kdtree')

| Model            | Fit Time (s)    | ARI vs sklearn | ARI vs fast_hdbscan | ARI vs truth |
|------------------|-----------------|----------------|---------------------|--------------|
| skl-contrib HDBS | 1.1947 ± 0.0133 |       -        |          -          | 0.9984       |
| fast hdbscan     | 0.2994 ± 0.0043 |       -        |          -          | 0.9984       |
| mojmelo HDBS     | 0.2041 ± 0.0060 | 0.9887         | 0.9984              | 0.9901       |

[`DBSCAN`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/dbs_bench.mojo) (algorithm='kd_tree')

| Model       | Fit Time (s)    | ARI vs sklearn | ARI vs truth |
|-------------|-----------------|----------------|--------------|
| sklearn DBS | 1.0625 ± 0.0020 |       -        | 0.8605       |
| mojmelo DBS | 0.4817 ± 0.0035 | 1.0000         | 0.8605       |

[`KNN`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/knn_bench.mojo) (algorithm='kd_tree')

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn KNN | 0.0353 ± 0.0005 | 1.7600 ± 0.0063  | 0.8543   |
| mojmelo KNN | 0.0149 ± 0.0006 | 0.2126 ± 0.0040  | 0.8347   |

[`SVM`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/svm_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn SVM | 1.2857 ± 0.0020 | 0.3720 ± 0.0008  | 0.9750   |
| mojmelo SVM | 0.8618 ± 0.0091 | 0.0600 ± 0.0002  | 0.9750   |

[`DecisionTreeClassifier`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/dtc_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn DTC | 0.9051 ± 0.0008 | 0.0004 ± 0.0000  | 0.9300   |
| mojmelo DTC | 0.0749 ± 0.0028 | 0.0002 ± 0.0000  | 0.9328   |

[`DecisionTreeRegressor`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/dtr_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | MSE       |
|-------------|-----------------|------------------|-----------|
| sklearn DTR | 0.6466 ± 0.0006 | 0.0005 ± 0.0000  | 8247.9358 |
| mojmelo DTR | 0.0795 ± 0.0049 | 0.0003 ± 0.0000  | 8192.1982 |

[`RandomForestClassifier`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/rfc_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn RFC | 0.4401 ± 0.0078 | 0.0139 ± 0.0002  | 0.9163   |
| mojmelo RFC | 0.4064 ± 0.0036 | 0.0044 ± 0.0001  | 0.9144   |

[`RandomForestRegressor`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/rfr_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | MSE       |
|-------------|-----------------|------------------|-----------|
| sklearn RFR | 2.0257 ± 0.0050 | 0.0134 ± 0.0004  | 8454.5517 |
| mojmelo RFR | 1.2247 ± 0.0094 | 0.0067 ± 0.0002  | 9155.6895 |

[`PCA`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/pca_bench.mojo) (svd_solver='full')

| Model       | Fit Time (s)    | Transform Time (s) | Explained Var |
|-------------|-----------------|--------------------|---------------|
| sklearn PCA | 0.2231 ± 0.0008 | 0.0063 ± 0.0000    | 0.5329        |
| mojmelo PCA | 0.0760 ± 0.0011 | 0.0166 ± 0.0013    | 0.5329        |

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

Contributions can be done to the project in these 3 ways:
1. Applying improvements to the code and opening a Pull Request
2. Reporting a bug
3. Suggesting new features

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* Mojo usage and distribution are licensed under the [Modular Community License](https://www.modular.com/legal/community).

* <a href='https://www.csie.ntu.edu.tw/~cjlin/libsvm/'>Libsvm</a>, A Library for Support Vector Machines by Chih-Chung Chang and Chih-Jen Lin licensed under the BSD-3-Clause license.

* `HDBSCAN` implementation is partially based on <a href='https://hdbscan.readthedocs.io/en/latest/'>hdbscan</a> by Leland McInnes, John Healy and Steve Astels licensed under the BSD-3-Clause license and <a href='https://fast-hdbscan.readthedocs.io/en/latest/'>Fast Multicore HDBSCAN</a> by Tutte Institute for Mathematics and Computing licensed under the BSD-2-Clause license.

* `matmul` implementation is based on <a href='https://github.com/YichengDWu/matmul.mojo'>matmul.mojo</a> by Ethan Wu (YichengDWu) licensed under the Apache-2.0 license.

* `argmin`, `argmax` and `argsort` implementations and `utils.algorithm` submodule are based on codes from <a href='https://github.com/modular/modular'>Modular</a> licensed under the Apache License v2.0 with LLVM Exceptions.

* <a href='https://arxiv.org/abs/physics/0408067'>KDTREE2</a>, a kd-tree implementation in Fortran 95 and C++ by Matthew B. Kennel.

* Initially drew inspiration from Patrick Loeber's <a href='https://github.com/patrickloeber/MLfromscratch/'>MLfromscratch</a>.


[issues-shield]: https://img.shields.io/github/issues/yetalit/mojmelo
[issues-url]: https://github.com/yetalit/mojmelo/issues
[license-shield]: https://img.shields.io/badge/license-BSD%203--Clause-blue
[license-url]: https://github.com/yetalit/Mojmelo/blob/main/LICENSE
