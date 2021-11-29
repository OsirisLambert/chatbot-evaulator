import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class fb_chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

    def chatbot(self, user_utterance):
        in_tensor = self.tokenizer.encode(user_utterance + self.tokenizer.eos_token, return_tensors='pt')
        if len(in_tensor[0])< 128:
            out_tensor = self.model.generate(in_tensor,
                                        #do_sample=True,
                                        max_length=128,
                                        #top_p=0.92,
                                        pad_token_id=self.tokenizer.eos_token_id) 
        else:
            s = torch.unsqueeze(in_tensor[0][-128:], dim=0)
            print(s)
            out_tensor = self.model.generate(s,
                                        #do_sample=True,
                                        max_length=128,
                                        #top_p=0.92,
                                        pad_token_id=self.tokenizer.eos_token_id)
        out_utterance = self.tokenizer.decode(out_tensor[0], skip_special_tokens=True)
        return out_utterance


if __name__ == '__main__':
    df = pd.read_csv('dailydialog_emotion_test.csv') # You will need to use your own path

    chatbot('You\'re welcome . We wish you a speedy recovery . Goodbye .')

    df_test = df.sample(20).drop(columns=['label'])
    df_test['sentence2'] = df_test.sentence1.apply(chatbot)
    df_test


    df_test.to_csv('dailydialog_emotion_test_response.csv')



