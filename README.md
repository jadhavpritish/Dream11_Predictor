## Dream11_Predictor

### HOWTo run the script locally - 

* Make sure that you have the data file in the data folder. The file columns must match with that of the 
sample file included in the github repo. 

* Setup the virtual environment on your local machine using pipenv. 
`pipenv install`--dev

Note- If you dont have pipenv installed on you local machine, first make sure to install it using - 
`sudo pip install pipenv`

* Once the virtualenv is installed, you can run the script using command- 
`pipenv run python src/player_selection_dream11.py`--player_data_filepath <absolute_data_filepath>
