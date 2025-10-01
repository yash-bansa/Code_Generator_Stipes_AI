# Code_Generator_Stipes_AI
Agentic AI Application for code generator usecase

## Reference repo

This repository has a folder called `examples` which function as the code repo.
Queries can be asked on the files which are in the examples repository.

## Installation process

1. Create a virtual environment (python version > 3.10).
2. Activate the created virtual environment.
3. pip install -r requirements.txt

## .env file must be included

Ask to development team for the .env file. There is already an example file (.env_example.txt) for your own keys.


## To run the backend

```bash
# Make sure your virtual environment is activated
python -m uvicorn main:app --reload
```

## To run the streamlit application

1. make sure that the backend is running.
2. open a new terminal, activate the virtual environment.

```bash
# Make sure your virtual environment is activated
streamlit run app_streamlit.py
```


## Sample queries

    After we provide a query and run it, the master planner will ask whether to proceed or not.
    If we wish to proceed, then just type yes.
    If no then say no and provide what updates u require.

1. Code generation

    a. Update the feature engineering logic in my code to perform categorical encoding for categorical data and replace null values with the most repeating feature.
    - yes


    b. In my validation code, add logic to filter out outliers.
    - yes


    c.  in the read_csv python file after reading the file mentioned, make sure to add a column called index_col which is self incrementing.
    - yes
    - In addition to that add one more column called is_supervision_required and set it to True.
    - yes


    d. in the read_csv python file after reading the file mentioned, make sure to add a column called index_col which is self incrementing.
    - no. In addition to that add one more column called is_supervision_required and set it to True.
    - yes

2. Code migration

    a. I need to migrate transform.py to pyspark version 3.5
    - yes

    b. Migrate all the files in my repo to pyspark version 3.5
    - yes

3. Config generation

    a. update the data transformation logic in sample_config.json by replacing the pass mark to 35
    - yes