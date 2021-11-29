import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class chatbot:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def generate_response(self, user_utterance):
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





