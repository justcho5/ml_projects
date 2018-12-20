# Machine Learning Projects

## Project 2: Project Recommender System

### Folder structure

```
project2/                              # Contains all project files
├── data/                              # contains the train and tests data
└── notebooks/                         # Contains Jupyter notebook files.
└── script/
    └── experimental/                  # contains file which were used to test different aproaches ( or started to test)
    ├── dataset.py
    ├── gridsearch.py
    └── matrix_factorization.py
    └── models.py
    └── plot_helper.py
    └── run.py
    └── run_all_blending.py
    └── run_all_models_one_by_one.py
    └── submission.py
```

#### Script file structure

**run.py**
This is the main file, which will reproduce our results.
It contains all the weights hard-coded with the best parameters for the model.

**dataset.py**
Contains helper function which are used to transfor the surprise strucutre
to marix form or the given submission file to pandas data frames.

**gridsearch.py**
Helper file which was used to perform grid serach on specific models and
store the results into a file.

**models.py**
Contains the surprise models we use and our own models.
Also includes helper class to perform cross validation.

**matrix_factorization.py**
Contains the matrix factorization method we looked into class.

**plot_helper.py**
Helper file which is used in the jupyter notebook to create plots

**run_all_blending.py**
Runs all modesl and try to find best blending weights.

**run_all_models_one_by_one.py**
Runs all the models one by one and save the results into a file.


**submission.py**
Helper file which contains code to create the submission file for CrowdAI.


### Requirements

The python code has been tested with the following versions:

```
Python: 3.6.2
Numpy: 1.13.1
pandas: 0.23.4
scipy: 1.1.0
scikit-surprise: 1.0.6
tqdm: 4.28.1
```

In order to generate the prediction file simply run the run.py script:

The test.csv and train.csv must be provided in project2/data/

    python run.py

This will create the output file in *project2/script/reproducable_submission_prediction.csv*

### Authors
Cho Hyun Jii
Poopalasingam Kirusanth
Rodriguez Natali



