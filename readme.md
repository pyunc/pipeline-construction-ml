## Table of contents

* [General info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General info
This tiny project aims to consolidate my coding skills to deploy traditional ML models passing through 3 different types of ML pipeline patterns. This project puts efforts in the software engineering best practices, patterns and design, though correct problem framing is still essential. 

* Procedural programming
* Object oriented programming
* Third part pipeline pattern (Sklearn)
	
## Dataset and Technologies

The dataset explores insurance costs and patients life style behavior. The data is available on Github [here](https://github.com/stedy/Machine-Learning-with-R-datasets) and [here](https://www.kaggle.com/mirichoi0218/insurance)


Project is created with:
* Jupyter notebook
* python 3.7
* sklearn
* pandas and numpy

## Evolution of the project

11-9-20

The **first project**  is based of procedural programming, which is a coding style i am more used it given my non original software engineering background. But still, this 'hackish' style is kind quickly to develop and a good to go for very small teams. Maintenance, robustness, extensability and a potencial coding smelling presence is very strong. 

This project is composed by 5 files. EXploration.ipynb is a jupyter notebook for quick exploration. Config.py gather all required parameters. Preprocessing.py is a list of functions, typical from ML development cycle. Train.py run all functions and parameters, dumping a pickled model. Finaly score.py measures the quality metrics.


Still running the project..