#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch


# In[2]:


df = pd.read_csv('dailydialog_emotion_test.csv') # You will need to use your own path


# In[3]:


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")


# In[4]:


def chatbot(user_utterance):
    in_tensor = tokenizer.encode(user_utterance + tokenizer.eos_token, return_tensors='pt')
    out_tensor = model.generate(in_tensor,
                                #do_sample=True,
                                max_length=1000,
                                #top_p=0.92,
                                pad_token_id=tokenizer.eos_token_id)
    out_utterance = tokenizer.decode(out_tensor[0], skip_special_tokens=True)
    return out_utterance


# In[5]:


chatbot('You\'re welcome . We wish you a speedy recovery . Goodbye .')


# In[9]:


df_test = df.sample(20).drop(columns=['label'])
df_test['sentence2'] = df_test.sentence1.apply(chatbot)
df_test


# In[11]:


df_test.to_csv('dailydialog_emotion_test_response.csv')


# In[ ]:




