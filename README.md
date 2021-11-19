# NameEntityTrainer

## Overview
This project is to train the labelled NER data and detect the NER in uploaded text file.
The main frameworks in this project are Flask and Spacy. 

## Structure

- src

    The source code for training and detection NER.
    
- utils

    * The models for NER model
    * The source code for the management of the folders and files of this project
    
- app

    The main execution file for running backend with Flask
    
- train

    The execution file for training
    
- requirements

    All the dependencies for this project
    
- settings

    The several settings including server setting, training json data path

## Installation

- Environment

    Windows 10, Ubuntu 18.04, Python 3.6

- Dependency Installation

    Please go ahead to this project directory and run the following command in the terminal.
    ```
        pip3 install -r requirements.txt
        python3 -m spacy download en_core_web_sm
    ``` 

## Execution

- Classification
    * Please run the following command in the terminal.

        ```
            python3 app.py
        ```
    * You can see server running at 5000 port.
    * The endpoint for detection of NER is "/api/detection" with POST request. 

- Training
    * Please set JSON_FILE_PATH as the full path of json file for training and EPOCHS as the number of training 
    iterations in settings file
    
    * Then, please run the following command in the terminal.
    ```
        python3 train.py
    ```  
