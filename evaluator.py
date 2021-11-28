from load_emotion import load_emotion
from load_neutral import load_non_neutral, load_neutral
import chatbot
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

if __name__ == '__main__':
	# Load dataset
	print("Reading Test File...")
	df = pd.read_csv('test1.csv', encoding='utf-8', error_bad_lines=False)

	# Load models
	# Neutral / Non-Neutral Classifier
	print("Loading Neutral / Non-Neutral Classifier...")
	nnc = pipeline('text-classification', model='Osiris/neutral_non_neutral_classifier')
	# Chatbot: facebook/blenderbot-400M-distill
	print("Loading Chatbot...")
	fb_chatbot = chatbot.fb_chatbot()
	# Emotion Classifier
	print("Loading Emotion Classifier...")
	ec = pipeline('text-classification', model='Osiris/emotion_classifier')

	# Step 1: Dropping out all neutral sentences
	print("Dropping out all neutral sentences...")
	non_neutral_df = load_non_neutral(nnc, df)

	# Step 2: Generate chatbot response, here, we use facebook/blenderbot-400M-distill as an example
	print("Generate chatbot response...")
	tqdm.pandas()
	non_neutral_df['sentence2'] = non_neutral_df.sentence1.progress_apply(fb_chatbot.chatbot)

	# Step 3: Checking emotional consistency, calculating socres
	print("Checking emotional consistency, calculating socres...")
	score = load_emotion(ec, nnc, non_neutral_df)
	print("chatbot evaluation score is: ", score)

