from transformers import PreTrainedTokenizerFast
import torch
from transformers import GPT2Config, GPT2LMHeadModel
import pandas as pd
import random



def find_longest_possible_tweet(generated_sequence):
    # Generated sequences are trimmed down to fit into the 280 characters limit of tweets
    longest_tweet = generated_sequence.split(".")[0]
    for sentence in generated_sequence.split(".")[1:]:
        candidate_tweet = longest_tweet + "." + sentence
        if len(candidate_tweet) > 279:
            return longest_tweet + "."
        else:
            longest_tweet = candidate_tweet

def generate_tweet(model, prompt):

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    # Model is applied to the incoming prompt
    generated_ids = model.generate(
        input_ids,
        max_length=300,
        do_sample=True,
        temperature=0.7
    )
    generated_sequence = tokenizer.decode(generated_ids[0])
    tweet = find_longest_possible_tweet(generated_sequence)

    return tweet

# previously identified common prompts are loaded
prompts = open('common_prompts.txt','r')
prompts = eval(prompts.read())

# The same tokenizer as during modelling is loaded
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../modelling/tokenizer.json")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# The model is loaded, using the same configurations as during modelling
sequence_length = 300
model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id,
    n_ctx=sequence_length,
    n_positions=sequence_length,
    n_head=8,
    n_layer=12,
    n_embd=512
)
model = GPT2LMHeadModel(model_config)
model.load_state_dict(torch.load("../modelling/trained_model.pt" ,map_location=torch.device('cpu')))

amount_tweets = 100
counter = 0
tweets_list = []

# generate tweets and collect them
while counter < amount_tweets:
    tweet = generate_tweet(model, random.choice(prompts))
    print(tweet)
    tweets_list.append(tweet)
    counter += 1

# Convert list to dataframe and save as csv
data = {'tweet':tweets_list}
df = pd.DataFrame(data)
df.to_csv("gpt2_pretweets_collection.csv")
