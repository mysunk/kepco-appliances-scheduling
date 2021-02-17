Smart appliance scheduling service for energy prosumer
=======================================
### 2020 전력데이터 신서비스 개발 경진대회
homepage: [link](https://home.kepco.co.kr/kepco/NS/main/main.do)

### __Structure__
The structure of the project is as follows.
```setup
.
└── main.py
└── test.py
└── plot-figs.py
└── utils.py
└── data
└── models
└── results
└── smp-modules
    └── smp-pred-ar.py
    └── smp-pred-lgb.py
    └── smp-pred-lstm.py
    └── smp-pred-util.py
└── ppt
```
* main.py: main file
* test.py: test file with pre-trained model in main file
* plot-figs.py: for visualization
* utils.py: utilities
* data: train and test data used in main and test file
* models: pre-trained models which is result of main file
* results: result of training
* smp-modules: smp prediction module
    - ar, lgb, lstm model
* ppt: presentation file

### __Context__
* [Introduction](#introduction)
* [Methodology](#methodology)
* [Result](#result)
* [Usage](#usage)
* [GUI dashboard](#gui-dashboard)
* [Prize](#prize)


Introduction
=======================

### Background
Will be updated soon

### Overview
![개요](img/overview.png)

Methodology
=======================
![method](img/method.png)

Result
=======================
Will be updated soon

Usage
==================
### Requirements 
To install requirements:
```setup
pip install -r requirements.txt
```
### Training and Evaluation
* Training
```train and eval
python main.py
```
* Evaluate (test)
```
python test.py
```

GUI dashboard
=======================
![gui](img/gui.JPG)

Prize
=======================
Got the grand prize   

Contact
==================
If there is something wrong or you have any questions, send me an e-mail or make an issue.  
[![Gmail Badge](https://img.shields.io/badge/-Gmail-d14836?style=flat-square&logo=Gmail&logoColor=white&link=mailto:pond9816@gmail.com)](mailto:pond9816@gmail.com)