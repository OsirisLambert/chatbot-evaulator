# chatbot-evaulator
In the current research, chatbots can recognize and acknowledge a speaker’s feelings in a conversation. However, we are still facing a big problem about how to ensure that chatbots are emotionally consistent. A practical and qualified chatbot can not only respond to human input but also will be able to cater to the emotional needs of the user. A chatbot is not supposed to respond “that’s great to hear” to the user’s input “I feel sick today”. Up to our knowledge, there is no existing way that can automatically evaluate the performance of a chatbot in terms of empathetic responding. In this research, we aim to train an evaluator that can effectively evaluate the emotional consistency of chatbots.

Check out our [paper](https://arxiv.org/abs/2112.01616).
# Load_neutral.py
In load_neutral / load_non_neutral function, it requires two parameters, one is a model and the other is a dataset. The main goal is to divide the sentences in the raw dataset into two groups, neutral and non-neutral ones. The load_neutral function will return a sub dataset that contains only neutral sentences from the original dataset while the load_non_neutral function will return a sub dataset that contains only non-neutral sentences. 

# chatbot.py
The only difference between generate_response and generate_response_gpt functions lies in the decoding: The Dialo-GPT-large returns an output tensor that concatenates the tensor corresponding to the user utterance and the tensor corresponding to the chatbot utterance, so the decoder in the generate_response_gpt function will first take the tensor corresponding to the chatbot utterance, and then decode it to get the chatbot utterance; while the Blenderbot-400M-distill returns a tensor corresponding to the chatbot utterance, so the decoder can directly decode it to get the chatbot utterance.
