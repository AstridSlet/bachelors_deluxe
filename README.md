# Bachelor deluxe

This repository (so far) contains all of the ML scripts for the paper on generalizability of machine-learning approaches in classifying autism spectrum disorders

## Running my scripts

For running my scripts I'd recommend doing the following from your terminal (and remembering to use the new environment that it creates):

__MAC/LINUX/WORKER02__
```bash
git clone https://github.com/emiltj/cds-visual.git
cd cds-visual
bash ./create_lang_venv.sh
```
__WINDOWS:__
```bash
git clone https://github.com/emiltj/cds-visual.git
cd cds-visual
bash ./create_lang_venv_win.sh
```

## Repo structure and files

This repository has the following directory structure:

| Column | Description|
|--------|:-----------|
```data```| Is meant to contain the testing and training data
```output``` | Contains the output of the ML process - dataframes, confusion matrices and classification reports

Furthermore it contains the files:
- ```./create_bach_venv.sh``` -> (Mac or Linux) A bash script which automatically generates a new virtual environment, and install all the packages contained within ```requirements.txt```
- ```./create_bach_venv.win.sh``` -> (Windows) A bash script which automatically generates a new virtual environment, and install all the packages contained within ```requirements.txt```
- ```requirements.txt``` -> A list of packages along with the versions that are required to run the scripts
- ```README.md``` -> This very readme file

## Contact

