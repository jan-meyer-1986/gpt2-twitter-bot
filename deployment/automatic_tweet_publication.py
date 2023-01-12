import tweepy
import time
import pandas as pd

# An external text file keeps count of the number of tweets published. The same number is used to index pretweet collection
with open('latest.txt') as f:
    latest_tweet_number = int(f.read())

print(latest_tweet_number)

df = pd.read_csv("gpt2_pretweets_collection.csv")

def tweet_now(msg):
    msg = msg[0:280]
    try:
        # Fill in valid Twitter credentials
        api_key=""
        api_key_secret = ""
        access_token = ""
        access_token_secret= ""

        # Authenticate to Twitter
        auth = tweepy.OAuthHandler(api_key, api_key_secret)
        auth.set_access_token(access_token, access_token_secret)
        api = tweepy.API(auth)
        try:
            api.verify_credentials()
            print("Authentication OK")
        except:
            print("Error during authentication")
        api = tweepy.API(auth, wait_on_rate_limit=True)
        # Actual tweeting takes place
        api.update_status(msg)
        print(msg)
        print("Tweeted")
    except Exception as e:
        print(e)

for idx, rows in df.iterrows():

    if idx <= latest_tweet_number:
        continue

    print(rows["tweet"])
    tweet_now(rows["tweet"])
    # Update external counter
    with open('latest.txt',"w") as f:
        f.write(str(idx))

    print("done")
    # Defer next tweet.
    time.sleep(24*3600)
