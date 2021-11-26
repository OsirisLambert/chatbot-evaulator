from transformers import pipeline
import pandas as pd

# https://huggingface.co/Osiris/neutral_non_neutral_classifier
# LABEL_1: Non Neutral (have some emotions)
# LABEL_0: Neutral (have no emotion)


def load_non_neutral(model, dataset):
    deleted = []
    for i in range(dataset.shape[0]):
        if model(dataset.iloc[i, 0])[0]['label'] == 'Neutral':
            deleted.append(i)
    dataset = dataset.drop(deleted)
    dataset.to_csv('non_neutral.csv', header=None, index=None)


def load_neutral(model, dataset):
    deleted = []
    for i in range(dataset.shape[0]):
        if model(dataset.iloc[i, 0])[0]['label'] == 'Non-Neutral':
            deleted.append(i)
    dataset = dataset.drop(deleted)
    dataset.to_csv('neutral.csv', header=None, index=None)


if __name__ == '__main__':
    data = pd.read_csv('sentence.csv', header=None)
    nnc = pipeline('text-classification', model='Osiris/neutral_non_neutral_classifier')
    load_neutral(nnc, data)
    load_non_neutral(nnc, data)


