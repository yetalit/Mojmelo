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

If you are not familiar with Mojo projects, you can get started here: https://docs.modular.com/mojo/manual/get-started/

### Prerequisites

* mojo-compiler 0.25.7

Additionally, you may want to install bellow Python packages for a better usability and to run tests:
1. Numpy
2. Pandas
3. Scikit-learn
4. Matplotlib

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

Note: If CPU cache details are available by your OS, benchmarking parts of the setup will be skipped. Otherwise, please try not to run other tasks on your pc during the process for better results.

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

Note: If CPU cache details are available by your OS, benchmarking parts of the setup will be skipped. Otherwise, please try not to run other tasks on your pc during the process for better results.

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

* Mojo usage and distribution are licensed under the [Modular Community License](https://www.modular.com/legal/community).

* <a href='https://www.csie.ntu.edu.tw/~cjlin/libsvm/'>Libsvm</a>, A Library for Support Vector Machines by Chih-Chung Chang and Chih-Jen Lin licensed under the BSD-3-Clause license.

* `matmul` implementation is based on <a href='https://github.com/YichengDWu/matmul.mojo'>matmul.mojo</a> by Ethan Wu (YichengDWu) licensed under the Apache-2.0 license. 

* `argmin`, `argmax` and `argsort` implementations are based on codes from <a href='https://github.com/modular/modular'>Modular</a> licensed under the Apache License v2.0 with LLVM Exceptions.

* <a href='https://arxiv.org/abs/physics/0408067'>KDTREE2</a>, a kd-tree implementation in Fortran 95 and C++ by Matthew B. Kennel.

* Initially drew inspiration from Patrick Loeber's <a href='https://github.com/patrickloeber/MLfromscratch/'>MLfromscratch</a>.


[issues-shield]: https://img.shields.io/github/issues/yetalit/mojmelo
[issues-url]: https://github.com/yetalit/mojmelo/issues
[license-shield]: https://img.shields.io/badge/license-BSD%203--Clause-blue
[license-url]: https://github.com/yetalit/Mojmelo/blob/main/LICENSE
