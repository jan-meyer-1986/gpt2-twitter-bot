# gpt2-twitter-bot

## Intro

This repository reflects the code base of my twitter bot at https://twitter.com/KI_Milsung.
The generative model is a GPT-2 and trains on a corpus consisting of 50 books
authored by the North Korean President Kim Il Sung found on https://www.marxists.org/archive/kim-il-sung/index.htm
That choice is based purely on entertainment grounds as well as on the ready availability 
of 45MB of training data. Absolutely no ideological endorsement is intended. 

While the entire code is bundled in this repository, its productive execution takes place 
on 3 different platforms. Since the model is computationally extremely intensive, GPU's on Google's Colab
are be employed.

The deployment of the bot itself takes place on the pythonanywhere platform. Since space there is limited, the model is 
not itself deployed there. Instead the trained model is used to
generate a list of tweets offline. That list is uploaded to pythonanywhere which triggers a tweet on a daily basis.

## Setup
In order to let model train, setup a virtual environment. 
````
conda create -n twitter_bot_env
source activate twitter_bot_env
conda install pip
pip install -r requirements.txt
````

## Usage

Download the corpus (e.g. https://www.marxists.org/archive/kim-il-sung/index.htm) and move the raw .txt
files to /data/all.
Run 
````
python3 modelling/data_preparation.py
python3 modelling/model_training.py
````

However, you are strongly advised to use a GPU.

Once the model is finished, create pre-tweets by running 
````
python3 model_application/tweet_generator.py
````
Make sure to play around with the temperature in model.generate(), as this controls the "creativity" of
the generator.

Once you created a csv files with pretweets, pick your platform of choice and deploy the bot to twitter 
using the code in the deployment folder.

