# chatbot-evaulator

# Load_neutral.py
In load_neutral / load_non_neutral function, it requires two parameters, one is a model and the other is a dataset. The main goal is to divide the sentences in the raw dataset into two groups, neutral and non-neutral ones. The load_neutral function will return a sub dataset that contains only neutral sentences from the original dataset while the load_non_neutral function will return a sub dataset that contains only non-neutral sentences. 

# chatbot.py
The only difference between generate_response and generate_response_gpt functions lies in the decoding: The Dialo-GPT-large returns an output tensor that concatenates the tensor corresponding to the user utterance and the tensor corresponding to the chatbot utterance, so the decoder in the generate_response_gpt function will first take the tensor corresponding to the chatbot utterance, and then decode it to get the chatbot utterance; while the Blenderbot-400M-distill returns a tensor corresponding to the chatbot utterance, so the decoder can directly decode it to get the chatbot utterance.
