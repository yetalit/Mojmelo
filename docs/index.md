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

If you are not familiar with Mojo projects, you can get started here: https://docs.modular.com/mojo/manual/get-started/

### Prerequisites

* mojo-compiler 0.26.2

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
rm -f ./setup"""
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

## Benchmarks (AMD Zen 4)

[`KMeans`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/kmeans_bench.mojo)

| Model          | Fit Time (s)    | ARI vs sklearn | ARI vs truth |
|----------------|-----------------|----------------|--------------|
| sklearn KMeans | 0.2782 ± 0.0010 |       -        | 0.9389       |
| mojmelo KMeans | 0.2276 ± 0.0005 | 1.0000         | 0.9389       |

[`HDBSCAN`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/hdbs_bench.mojo) (algorithm='boruvka_kdtree')

| Model            | Fit Time (s)    | ARI vs sklearn | ARI vs truth |
|------------------|-----------------|----------------|--------------|
| skl-contrib HDBS | 1.2791 ± 0.0042 |       -        | 0.9977       |
| mojmelo HDBS     | 0.3536 ± 0.0013 | 0.9828         | 0.9844       |

[`DBSCAN`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/dbs_bench.mojo) (algorithm='kd_tree')

| Model       | Fit Time (s)    | ARI vs sklearn | ARI vs truth |
|-------------|-----------------|----------------|--------------|
| sklearn DBS | 1.1968 ± 0.0083 |       -        | 0.8723       |
| mojmelo DBS | 0.5313 ± 0.0034 | 0.9999         | 0.8722       |

[`KNN`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/knn_bench.mojo) (algorithm='kd_tree')

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn KNN | 0.0357 ± 0.0011 | 1.6829 ± 0.0076  | 0.9227   |
| mojmelo KNN | 0.0117 ± 0.0004 | 0.2609 ± 0.0028  | 0.9104   |

[`SVM`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/svm_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn SVM | 1.8318 ± 0.0051 | 0.5775 ± 0.0006  | 0.9317   |
| mojmelo SVM | 1.1861 ± 0.0096 | 0.0956 ± 0.0031  | 0.9317   |

[`DecisionTreeClassifier`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/dtc_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn DTC | 0.9051 ± 0.0008 | 0.0004 ± 0.0000  | 0.9300   |
| mojmelo DTC | 0.0749 ± 0.0028 | 0.0002 ± 0.0000  | 0.9328   |

[`DecisionTreeRegressor`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/dtr_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | MSE       |
|-------------|-----------------|------------------|-----------|
| sklearn DTR | 0.6518 ± 0.0003 | 0.0006 ± 0.0000  | 1185.1717 |
| mojmelo DTR | 0.0664 ± 0.0023 | 0.0002 ± 0.0000  | 1175.6884 |

[`RandomForestClassifier`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/rfc_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | Accuracy |
|-------------|-----------------|------------------|----------|
| sklearn RFC | 0.4550 ± 0.0076 | 0.0138 ± 0.0006  | 0.9348   |
| mojmelo RFC | 0.4041 ± 0.0114 | 0.0064 ± 0.0001  | 0.9326   |

[`RandomForestRegressor`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/rfr_bench.mojo)

| Model       | Fit Time (s)    | Predict Time (s) | MSE       |
|-------------|-----------------|------------------|-----------|
| sklearn RFR | 2.0257 ± 0.0050 | 0.0134 ± 0.0004  | 8454.5517 |
| mojmelo RFR | 1.2247 ± 0.0094 | 0.0067 ± 0.0002  | 9155.6895 |

[`PCA`](https://github.com/yetalit/Mojmelo/blob/main/benchmarks/pca_bench.mojo) (svd_solver='full')

| Model       | Fit Time (s)    | Transform Time (s) | Explained Var |
|-------------|-----------------|--------------------|---------------|
| sklearn PCA | 0.2082 ± 0.0022 | 0.0062 ± 0.0000    | 0.5475        |
| mojmelo PCA | 0.0915 ± 0.0006 | 0.0261 ± 0.0009    | 0.5475        |

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

* `HDBSCAN` implementation is partially based on <a href='https://hdbscan.readthedocs.io/en/latest/'>hdbscan</a> by Leland McInnes, John Healy and Steve Astels licensed under the BSD-3-Clause license and <a href='https://fast-hdbscan.readthedocs.io/en/latest/'>Fast Multicore HDBSCAN</a> by Tutte Institute for Mathematics and Computing licensed under the BSD-2-Clause license.

* `matmul` implementation is based on <a href='https://github.com/YichengDWu/matmul.mojo'>matmul.mojo</a> by Ethan Wu (YichengDWu) licensed under the Apache-2.0 license.

* `argmin`, `argmax` and `argsort` implementations are based on codes from <a href='https://github.com/modular/modular'>Modular</a> licensed under the Apache License v2.0 with LLVM Exceptions.

* <a href='https://arxiv.org/abs/physics/0408067'>KDTREE2</a>, a kd-tree implementation in Fortran 95 and C++ by Matthew B. Kennel.

* Initially drew inspiration from Patrick Loeber's <a href='https://github.com/patrickloeber/MLfromscratch/'>MLfromscratch</a>.


[issues-shield]: https://img.shields.io/github/issues/yetalit/mojmelo
[issues-url]: https://github.com/yetalit/mojmelo/issues
[license-shield]: https://img.shields.io/badge/license-BSD%203--Clause-blue
[license-url]: https://github.com/yetalit/Mojmelo/blob/main/LICENSE
