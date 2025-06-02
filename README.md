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
* DBSCAN
* SVM
    1. Primal
    2. Dual
* Perceptron (single layer: Binary Classification)
* Naive Bayes
    1. GaussianNB
    2. MultinomialNB
* Decision Tree (both Regression/Classification)
* Random Forest (both Regression/Classification)
* GBDT (both Regression/Classification)
* PCA
* LDA
* Adaboost

Preprocessing:
* normalize
* MinMaxScaler
* StandardScaler
* KFold
* GridSearchCV

**Mojmelo will not only be limited to above algorithms.**

## Getting Started

If you are not familiar with Mojo projects, you can get started here: https://docs.modular.com/mojo/manual/get-started/

### Prerequisites

* MAX 25.3

Additionally, you may want to install bellow Python packages for a better usability and to run tests:
1. Pandas
2. Scikit-learn
3. Matplotlib

### Installation

There are two ways you can install mojmelo: Using Pixi CLI or through the source code.

To complete the installation, you should also finish the setup process which will be discussed later.

#### Pixi CLI

Make sure you have the Modular community channel (https://repo.prefix.dev/modular-community) in your `pixi.toml` file in the channels section, then Run the following command:
```
pixi add mojmelo
```

To start the setup process, Run the following command from the `main folder` of your project:
```
bash ./.pixi/envs/default/etc/conda/test-files/mojmelo/0/tests/setup.sh
```

Note: For better results, please try not to run other tasks on your pc during the setup process.

#### Source Code

You can also install mojmelo through the source code. This way, you will have the source code in your project.

First, Download `mojmelo` folder and `setup.mojo` file. Then Add the following task to your `pixi.toml` file in the tasks section:
```
[tasks]
setup = """
cd <path_to_mojmelo_location> &&
mojo ./setup.mojo &&
mojo ./setup.mojo 1 &&
mojo ./setup.mojo 2 &&
mojo ./setup.mojo 3 &&
mojo ./setup.mojo 4 &&
mojo ./setup.mojo 5 &&
mojo ./setup.mojo 6 &&
mojo ./setup.mojo 7 &&
mojo ./setup.mojo 8 &&
mojo ./setup.mojo 9"""
```

Don't forget to change `<path_to_mojmelo_location>` according to where `mojmelo` folder and `setup.mojo` file are stored.

Then Run the following command to start the setup process:
```
pixi run setup
```

Note: For better results, please try not to run other tasks on your pc during the setup process.

## Usage

Just import any model you want this way:
```python 
from mojmelo.LinearRegression import LinearRegression
```
You may also want to use the utility codes I've written for this project:
```python 
from mojmelo.utils.Matrix import Matrix
from mojmelo.utils.utils import *
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

You can contribute to the project in 3 ways:
1. Apply improvements to the code and Open a Pull Request
2. Report a bug
3. Suggest new features

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Took inspiration from Patrick Loeber's <a href='https://github.com/patrickloeber/MLfromscratch/'>MLfromscratch</a> and Erik Linder-Norén's <a href='https://github.com/eriklindernoren/ML-From-Scratch/'>ML-From-Scratch</a>

MAX and Mojo usage and distribution are licensed under the [Modular Community License](https://www.modular.com/legal/community).


[issues-shield]: https://img.shields.io/github/issues/yetalit/mojmelo
[issues-url]: https://github.com/yetalit/mojmelo/issues
[license-shield]: https://img.shields.io/badge/license-BSD%203--Clause-blue
[license-url]: https://github.com/yetalit/Mojmelo/blob/main/LICENSE
