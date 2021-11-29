from load_emotion import load_emotion
from load_neutral import load_non_neutral, load_neutral
import chatbot
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Evaluator')
	parser.add_argument('-f', '--file', type=str, metavar='', default='test.csv',
	                    help='file path. Default: %(default)s')
	# parser.add_argument('-n', '--nnc', action='store_false',
	#                     help='Neutral / Non-Neutral Classifier')
	# parser.add_argument('-e', '--ec', action='store_false',
	#                     help='Emotion Classifier')
	parser.add_argument('-hu', '--human_eval', action='store_true',
	                    help='step4 Human evaluation score')

	args = parser.parse_args()

	# Load dataset
	print("Reading Test File...")
	df = pd.read_csv(args.file, encoding='utf-8', error_bad_lines=False, skip_blank_lines=True)

	# Load models
	# Neutral / Non-Neutral Classifier
	print("Loading Neutral / Non-Neutral Classifier...")
	nnc = pipeline('text-classification', model='Osiris/neutral_non_neutral_classifier')
	# Chatbot: facebook/blenderbot-400M-distill
	print("Loading Chatbot...")
	fb = chatbot.fb_chatbot()
	# Emotion Classifier
	print("Loading Emotion Classifier...")
	ec = pipeline('text-classification', model='Osiris/emotion_classifier')

	# Step 1: Dropping out all neutral sentences
	print("Dropping out all neutral sentences...")
	non_neutral_df = load_non_neutral(nnc, df)

	# Step 2: Generate chatbot response, here, we use facebook/blenderbot-400M-distill as an example
	print("Generate chatbot response...")
	tqdm.pandas()
	non_neutral_df['sentence2'] = non_neutral_df.sentence1.progress_apply(fb.chatbot)

	# # Step 3: Checking emotional consistency, calculating socres
	# print("Checking Chatbot emotional consistency, calculating socres...")
	# c_score = load_emotion(ec, nnc, non_neutral_df)
	# print("chatbot evaluation score is: ", c_score)

	if args.human_eval:
		# Step 4: (OPTIOANL) Check human response, calculating socres
		print("Checking Huamn response emotional consistency, calculating socres...")
		h_score = load_emotion(ec, nnc, non_neutral_df, 'Response')
		print("huamn evaluation score is: ", h_score)

