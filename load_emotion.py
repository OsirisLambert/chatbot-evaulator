# nnc label: Non-Neutral  Neutral
# ec label: Positive  Negative

import pandas as pd
from transformers import pipeline

csvFile = "test.csv"   # csv file name


def load_emotion(ec_model, nnc_model, dataset, compare_col_name='sentence2'):
	score = 0
	ct = 0
	for idx, row in dataset.iterrows():
		label_nn = nnc_model(row[compare_col_name])[0]['label']	# row['sentence2'] ==> system response
		# if system utterance is neutral => get 0.5 point
		if label_nn == 'Neutral':
			score += 0.5
		else:
			label_user = ec_model(row['sentence1'])[0]['label']	# row['sentence1'] ==> user response
			label_sys = ec_model(row[compare_col_name])[0]['label']
			# if user and system are both pos/neg => get 1.0 point
			if label_user == label_sys:
				score += 1
			# if not => get 0 point
		ct += 1
	return score/ct


# This function can calculate dialog level performance.
def check_dialog_level_score(compare_column_name):
    max_total = []
    min_total = []
    conver = []
    for idx, row in df.iterrows():
        try:
            label_nn = nnc(row[compare_column_name])[0]['label']
            if label_nn == 'Neutral':
                score = 0.5
            else:
                label_user = ec(row['user'])[0]['label']# row['sentence1'] ==> user response
                label_sys = ec(row[compare_column_name])[0]['label']
                # if user and system are both pos/neg => get 1.0 point
                if label_user == label_sys:
                    score = 1
                else:
                    score = 0
            conver.append(score)
        except:
            max_total.append(max(conver))
            min_total.append(min(conver))
            conver = []
            continue
    return sum(max_total) / len(max_total), sum(min_total) / len(min_total)


if __name__ == '__main__':
	dataset = pd.read_csv(csvFile, header=0)
	ec = pipeline('text-classification', model='Osiris/emotion_classifier')
	nnc = pipeline('text-classification', model='Osiris/neutral_non_neutral_classifier')
	score = load_emotion(ec, nnc, dataset)
	print("%.2f" % score)

