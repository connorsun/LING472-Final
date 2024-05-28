from os import listdir, scandir
from os.path import isfile, join
import numpy as np
from naive_bayes import NaiveBayes
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class NaiveBayesBasic(NaiveBayes):
    def __init__(self):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}
        self.tokenizer = None
        self.model = None
        self.spam_centroid = None
        self.ham_centroid = None

    def get_embedding(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Using mean of last layer hidden states as the sentence embedding
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def calculate_centroid(self, embeddings):
        return np.mean(embeddings, axis=0)

    def train(self):
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        SPAM_FILES = []
        HAM_FILES = []
        TESTING_PATH = "./Training"
        TESTING_FOLDERS_PATHS = [f.path for f in scandir(TESTING_PATH) if f.is_dir()]
        #for testing_folder_path in TESTING_FOLDERS_PATHS:
        testing_folder_path = TESTING_FOLDERS_PATHS[0]
        TEST_SPAM_PATH = join(testing_folder_path, "spam")
        TEST_HAM_PATH = join(testing_folder_path, "ham")
        SPAM_FILES += [join(TEST_SPAM_PATH, file) for file in listdir(TEST_SPAM_PATH) if
                       isfile(join(TEST_SPAM_PATH, file))]
        HAM_FILES += [join(TEST_HAM_PATH, file) for file in listdir(TEST_HAM_PATH) if
                      isfile(join(TEST_HAM_PATH, file))]
        ham_total = 0
        ham_freq = {}
        spam_total = 0
        spam_freq = {}
        for ham_file in HAM_FILES:
            file = open(ham_file, "r", encoding="ISO-8859-1")
            for line in file:
                tokens = line.strip().split()
                ham_total += len(tokens)
                for token in tokens:
                    ham_freq[token] = ham_freq.setdefault(token, 0) + 1
            file.close()
        for spam_file in SPAM_FILES:
            file = open(spam_file, "r", encoding="ISO-8859-1")
            for line in file:
                tokens = line.strip().split()
                spam_total += len(tokens)
                for token in tokens:
                    spam_freq[token] = spam_freq.setdefault(token, 0) + 1
            file.close()
        self.ham_total = float(ham_total)
        self.ham_freq = ham_freq
        self.spam_total = float(spam_total)
        self.spam_freq = spam_freq

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")

        spam_words = list(spam_freq.keys())
        ham_words = list(ham_freq.keys())
        batch_size = 1024

        spam_embeddings = []
        for i in range(0, len(spam_words), batch_size):
            batch = spam_words[i:i + batch_size]
            embeddings = self.get_embedding(batch)
            spam_embeddings.extend(embeddings)

        ham_embeddings = []
        for i in range(0, len(ham_words), batch_size):
            batch = ham_words[i:i + batch_size]
            embeddings = self.get_embedding(batch)
            ham_embeddings.extend(embeddings)

        self.spam_centroid = self.calculate_centroid(spam_embeddings)
        self.ham_centroid = self.calculate_centroid(ham_embeddings)



    def predict(self, filename):
        file = open(filename, "r", encoding="ISO-8859-1")
        new_embedding = self.get_embedding(file.read())
        file.close()

        distance_to_spam = np.linalg.norm(new_embedding - self.spam_centroid)
        distance_to_ham = np.linalg.norm(new_embedding - self.ham_centroid)
        return distance_to_spam >= distance_to_ham



if __name__ == "__main__":
    nb = NaiveBayesBasic()
    nb.train()
    accuracy = nb.test()
    print(accuracy)