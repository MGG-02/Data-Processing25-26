import os
import numpy as np 
import pandas as pd
import json
import matplotlib.pyplot as plt
from collections import Counter
from rich.console import Console

from word_cleanner import *

from convert_annotations_data import *

seed = 42
np.random.seed(seed)

texts = []
fav_counts = []
retweet_counts = []
date = []
username = []
account_date = []
protected = []
verified = []
followers = []
followings = []
tweets_count = []
hashtag = []
url = []
events = []
y = []

# navigate the folders to create our dataset. In our task we only use the source tweets that are rumours
for f in folds:
  path1 = os.path.join('all-rnr-annotated-threads', f, 'rumours')
  # path1 = os.path.join('pheme-rumour-scheme-dataset', 'threads', 'en', f)
  for dir1 in os.listdir(path1):
        if '_' not in dir1:
          path_target  = os.path.join(path1,dir1,'annotation.json')
          file = open(path_target)
          data = json.load(file)
          target = convert_annotations_data(data)
          y.append(target)
          path2 = os.path.join(path1, dir1,'source-tweets')
          for dir2 in os.listdir(path2):
            if '_' not in dir2:
              path3  = os.path.join(path2,dir2)
              file = open(path3)
              data = json.load(file)
            
              #tweet features
              text = data['text']
              tweet_date = data['created_at']
              fav = data['favorite_count']
              retw = data['retweet_count']
                
              #user features
              usernames = data['user']['screen_name']
              account_creation = data['user']['created_at']
              is_protected = data['user']['protected']
              is_verified = data['user']['verified']
              no_followers = data['user']['followers_count']
              no_followings = data['user']['friends_count']
              no_tweets = data['user']['statuses_count']
                
              #entities
              no_hashtags = len(data['entities']['hashtags'])      
              has_url = data['entities']['urls']  
              text = data['text']
              fav = data['favorite_count']
              retw = data['retweet_count']
              
              texts.append(text)
              date.append(tweet_date)
              fav_counts.append(fav)
              retweet_counts.append(retw)
                                     
              username.append(usernames)
              account_date.append(account_creation)
              protected.append(is_protected)
              verified.append(is_verified)
              followers.append(no_followers)
              followings.append(no_followings)
              tweets_count.append(no_tweets)
            
              
              hashtag.append(no_hashtags)
              url.append(has_url)
            
              events.append(f)

console = Console()

df = pd.DataFrame([texts,date,fav_counts,retweet_counts,username,account_date,followers,followings,tweets_count,protected,verified,hashtag,url,events,y],['text','date','fav_count','retweet_count','username','account_date','followers','followings','tweet_count','protected','verified','no_hashtags','urls','event','target']).transpose()
df = df.infer_objects()


# drop categorical data and protected which has 0 var
df.drop(["date","username","account_date","urls","protected"], axis=1, inplace=True)

print('')
console.print("[bold cyan]Dataset shape[/bold cyan]")
console.print(df.shape)

#convert boolen features into numerical
df = df.astype({"verified":'int64'})
print('')
console.print("[bold magenta]Data types[/bold magenta]")
console.print(df.dtypes)
print('')
console.print("[bold cyan]### --- After deleting non symbolic features --- ###[/bold cyan]")
console.print(df.sample(frac=1).head())
print('')
console.print("[bold magenta]Missing values[/bold magenta]")
console.print(df.isna().sum())

print('')
console.print("[bold cyan]Basic statistics for numerical features[/bold cyan]")
console.print(df.describe())

print('')
console.print("[bold magenta]Target distribution[/bold magenta]")
console.print(df['target'].value_counts())

print('')
all_words = []

for t in df["text"]:
    all_words.extend(clean_words(str(t)))

word_freq = Counter(all_words).most_common(10)

console.print("[bold cyan] Top 10 Most Frequent Words [/bold cyan]")
console.print(word_freq)

from wordcloud import WordCloud

wc = WordCloud(width=1200, height=600, background_color="white").generate(" ".join(all_words))
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Tweets")
plt.show()

console.print("[bold magenta]### Example Tweets by Class ###[/bold magenta]")

for label in ["true", "false", "unverified"]:
    console.print(f"\n[bold yellow]{label.upper()}[/bold yellow]")
    console.print(df[df["target"] == label].sample(1)["text"].values[0])
