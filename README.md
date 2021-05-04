# Bachelor deluxe
This repository (so far) contains all of the ML scripts for the paper on generalizability of machine-learning approaches in classifying autism spectrum disorders

## Running the scripts
For running the scripts I'd recommend doing the following from your terminal (and remembering to use the new environment that it creates):

__MAC/LINUX
```bash
git clone https://github.com/emiltj/bachelors_deluxe.git
cd bachelors_deluxe
bash ./create_bach_venv.sh
```
__WINDOWS:__
```bash
git clone https://github.com/emiltj/bachelors_deluxe.git
cd bachelors_deluxe
bash ./create_bach_venv_win.sh
```

## Repo structure and files
This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| Is meant to contain the training and the testing data, a long with the feature lists and the models when the models have been trained
```predictions``` | Contains the output of the ML process - dataframes, confusion matrices and classification reports

Furthermore it contains the files:
- ```./create_bach_venv.sh``` -> (Mac or Linux) A bash script which automatically generates a new virtual environment, and install all the packages contained within ```requirements.txt```
- ```./create_bach_venv.win.sh``` -> (Windows) A bash script which automatically generates a new virtual environment, and install all the packages contained within ```requirements.txt```
- ```requirements.txt``` -> A list of packages along with the versions that are required to run the scripts
- ```README.md``` -> This very readme file

## Contact

