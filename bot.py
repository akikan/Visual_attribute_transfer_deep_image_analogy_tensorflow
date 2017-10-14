# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import os
import glob
from tensorflow.contrib import learn
import tweepy
from urllib.request import urlopen
import time
import VisualAttributeTransfer as VAT

CK = ""
CS = ""
AT = ""
AS = ""
Twitter_account=""#@からはじまるやつ 
#例) Twitter_account="@hogehoge"

# Twitterオブジェクトの生成
auth = tweepy.OAuthHandler(CK, CS)
auth.set_access_token(AT, AS)
api = tweepy.API(auth)




class Listener(tweepy.StreamListener):
    sess = tf.InteractiveSession()
    NUM_CLASSES = 11
    IMAGE_SIZE = 48
    CHANNEL = 3
    IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*CHANNEL
    


    def on_status(self, status):
        try:
            status_id = status.id
            screen_name = status.author.screen_name
            tweet_part = status.text.split(Twitter_account+" ")
            
            if status.extended_entities['media']!=[]:
                if status.in_reply_to_screen_name == Twitter_account or len(tweet_part) != 1:
                    pathes = []
                    for media in status.extended_entities['media']:
                        url = media['media_url_https']
                        url_orig = '%s:orig' %  url
                        filename = url.split('/')[-1]
                        savepath = "./" + filename
                        
                        response = urlopen(url_orig)
                        with open(savepath, "wb") as f:
                            f.write(response.read())
                        pathes.append(savepath)
                    
                    ret_pathes = VAT.run(pathes)
                    screen=status.author.screen_name
                    api.update_with_media(filename=ret_pathes[0],in_reply_to_status_id=status_id)
                    api.update_with_media(filename=ret_pathes[1],in_reply_to_status_id=status_id)


        except Exception as e:
            print("[-] Error: ", e)
        return True
     
    def on_error(self, status_code):
        print('Got an error with status code: ' + str(status_code))
        return True
     
    def on_timeout(self):
        print('Timeout...')
        return True
 


#参考サイト
#userstreamの使いかた
# http://ha1f-blog.blogspot.jp/2015/02/tweepypythonpip-tweepymac.html

    


try:
    listener = Listener()
    stream = tweepy.Stream(auth, listener)
    stream.userstream()
except:
    pass