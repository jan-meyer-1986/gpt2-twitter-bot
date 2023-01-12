# gpt2-twitter-bot

## Intro

This repository reflects the code base of my twitter bot. 
The generative model is a GPT-2 and trains on a corpus consisting of 50 books
authored by the North Korean President Kim Il Sung found on https://www.marxists.org/archive/kim-il-sung/index.htm
That choice is based purely on entertainment grounds as well as on the ready availability 
of 45MB of training data. Absolutely no ideological endorsement is intended. 

While the entire code is bundled in this repository, its productive execution takes place 
on 3 different platforms. Since the model is extremely computational intensive, GPU's on Google's Colab
are be employed.

The deployment of the bot itself takes place on the pythonanywhere platform. Since space there is limited, the model is 
not itself deployed there. Instead the trained model is used to
generate a list of tweets. That list is uploaded to pythonanywhere which triggers a tweet on a daily basis.


