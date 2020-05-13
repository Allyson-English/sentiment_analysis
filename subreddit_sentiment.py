import numpy as np
import textblob as textblob
import praw
from datetime import datetime
import matplotlib.pyplot as plt
import json
import creds
from pprint import pprint
import pandas as pd
from time import time

#Importing NLTK library and associated packaged

import nltk
nltk.__version__
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Importing textblob to compare sentiment analysis results with those from nltk

from textblob import TextBlob

# Reddit API wrapper 

reddit = praw.Reddit(client_id=creds.client_id, \
                     client_secret=creds.client_secret, \
                     user_agent=creds.user_agent, \
                     username=creds.username, \
                     password=creds.password)

#pulls all discussion ids available and adds them to existing dictionary, best used for starting new dataset or updating after a few days

def get_disc_ids(subreddit, existing_dict_w_all_data):
    
    discussion_ids = []
    for submission in subreddit.search('Daily Discussion Post'):
        if 'Daily Discussion Post' in submission.title:
            discussion_ids.append(submission.id)
      
    for d in discussion_ids:
        if d not in existing_dict_w_all_data.keys():
            print("Added", d, "to discussion dictionary.")
            existing_dict_w_all_data[d] = {}
        else: 
            pass
    
    if len(discussion_ids) != len(existing_dict_w_all_data):
        print("Alert! Length of discussion_ids does not equal length of dictionary.")
    
    return existing_dict_w_all_data

#checks for daily discussion ID. Best for updating dataset within same day or adding a single day 

def todays_disc_id(NAME_OF_SUBREDDIT):
    subreddit = reddit.subreddit(NAME_OF_SUBREDDIT)
    discussion_id = str(list(subreddit.hot(limit=1)))
    final_id = discussion_id.replace("[Submission(id='","").replace("')]","")
    return(final_id)

#pulls comment information, including comment body and sentiment, into a dictionary and returns dictionary 

def comment_info(us_com, submission, sid):
    
    token_dict = {}
    
    comment = us_com.body
    comment = comment.replace('\n', ' ')
    comment = comment.replace('I\'m', 'i am').replace('i\'m', 'i am').replace('i\'ll', 'i will').replace('I\'ll', 'i will')
    comment = comment.lower()
    
    #grab date/ time info for each comment 
    utc = submission.created_utc
    dt_object = datetime.fromtimestamp(utc)  

    #performing sentiment analysis
    ss = sid.polarity_scores(comment)
    
    if comment != '[removed]':
        token_dict.update({"comment_body" : comment})
        token_dict.update({"month":dt_object.strftime("%B")})
        token_dict.update({"day" : dt_object.strftime("%d")})
        token_dict.update(ss)
    
    return token_dict


#pulls everything together and returns complete dataset in the form of a dictionary

def subreddit_sentiment(discussion_ids, existing_dict, reddit = reddit):

    sid = SentimentIntensityAnalyzer()
    st = time()
    dictionary = {}
    
    for i in discussion_ids:
        print(i)
        dictionary[i] = {}
        submission = reddit.submission(i)
        submission.comments.replace_more(limit=0)

        for user_comment in submission.comments:
            tok_dict = comment_info(user_comment, submission, sid)

            if str(user_comment) not in existing_dict.keys():
                dictionary[i].setdefault(str(user_comment),tok_dict)
    
    
    print("\nProcessing time:", round((time()-st)/60, 2), "minutes.")
    
    return dictionary