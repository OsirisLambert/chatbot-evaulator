# nnc label: Non-Neutral  Neutral
# ec label: Positive  Negative

import pandas as pd
from transformers import pipeline

csvFile = "test.csv"   # csv file name


def load_emotion(ec_model, nnc_model, dataset):
	score = 0
	ct = 0
	for idx, row in dataset.iterrows():
		label_nn = nnc(row['System'])[0]['label']
		# if system utterance is neutral => get 0.5 point
		if label_nn == 'Neutral':
			score += 0.5
		else:
			label_user = ec(row['User'])[0]['label']
			label_sys = ec(row['System'])[0]['label']
			# if user and system are both pos/neg => get 1.0 point
			if label_user == label_sys:
				score += 1
			# if not => get 0 point
		ct += 1
	return score/ct

if __name__ == '__main__':
	dataset = pd.read_csv(csvFile, header=0)
	ec = pipeline('text-classification', model='Osiris/emotion_classifier')
	nnc = pipeline('text-classification', model='Osiris/neutral_non_neutral_classifier')
	score = load_emotion(ec, nnc, dataset)
	print("%.2f" % score)

