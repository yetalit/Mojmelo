<a id="readme-top"></a>

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![BSD-3-Clause License][license-shield]][license-url]

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
* SVM
    1. Primal
    2. Dual
* Perceptron (single layer: Binary Classification)
* Naive Bayes
    1. GaussianNB
    2. MultinomialNB
* Decision Tree (both Regression/Classification)
* Random Forest (both Regression/Classification)
* PCA
* LDA
* Adaboost

**The plan is to implement more ML algorithms including neural networks.**

## Getting Started

The following steps let you know how to get started with Mojmelo.

### Prerequisites

* Mojo compiler

Additionally, you may want to install bellow Python packages for a better usability and to run tests:
1. Numpy
2. Pandas
3. Scikit-learn

### Installation

You can put the compiled version of the package into your project and use it. The file can be found in `releases` folder.

or just download the whole `mojmelo` folder and put it into your project.

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

Inspired from Patrick Loeber's <a href='https://github.com/patrickloeber/MLfromscratch/'>MLfromscratch</a>


[stars-shield]: https://img.shields.io/github/stars/yetalit/mojmelo?style=social
[stars-url]: https://github.com/yetalit/mojmelo/stargazers
[issues-shield]: https://img.shields.io/github/issues/yetalit/mojmelo
[issues-url]: https://github.com/yetalit/mojmelo/issues
[license-shield]: https://img.shields.io/badge/license-BSD%203--Clause-blue
[license-url]: https://github.com/yetalit/Mojmelo/blob/main/LICENSE