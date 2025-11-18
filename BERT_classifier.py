import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
from load_dataset import *

#just keep text and targets for BERT
labels_to_drop = ['event',
                  'thread_id',
                  'tweet_id',
                  'date',
                  'fav_counts',
                  'retweet_counts',
                  'username',
                  'account_date',
                  'protected',
                  'verified',
                  'followers',
                  'friends_count',
                  'followings',
                  'hashtag',
                  'urls']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

data_path = 'csv_Dataset.csv'
pheme_df = pd.read_csv(data_path).drop(labels_to_drop, axis=1)

