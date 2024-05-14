# Disentangling Domain and General Representations for Time Series Classification

## About

This project is the implementation of paper "Disentangling Domain and General Representations for Time Series Classification"

## Dependencies

This project is implemented primarily in Python 3.9.16. Some other Python module dependencies are listed in ```requirements.txt```, which can be easily installed with pip:
  ```
  pip install -r requirements.txt
  ```

## File folders

`dataset`: dataset.py for generate dataset.

`scripts`: some quick starts of ucihar and wisdm dataset.

`model`: model of CADT

`utils`: data argumentation of CADT

## Run

Before building the project, we recommend switching the working directory to the project root directory. Assume the project root is at ``<dynamic_triad_root>``, then run command
```
cd <dynamic_triad_root>
```
Note that we assume ``<dynamic_triad_root>`` as your working directory in all the commands presented in the rest of this documentation.Then you can run bash script/ucihar_run.sh to have a quick start.
