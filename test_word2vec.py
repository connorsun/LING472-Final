from os import listdir, scandir
from os.path import isfile, join
import numpy as np
from naive_bayes import NaiveBayes
from naive_bayes_basic import NaiveBayesBasic
from gensim.models import Word2Vec
# from gensim.models.keyedvectors import WordEmbeddingsKeyedVectors
# import gensim
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class EmbeddingBayes(NaiveBayes):
    def __init__(self):
        self.ham_total = 0
        self.ham_freq = {}
        self.spam_total = 0
        self.spam_freq = {}
        # self.tokenizer = None
        # self.model = None
        # self.spam_centroid = None
        # self.ham_centroid = None
        self.spam_embeddings = None
        self.ham_embeddings = None
        self.nbb = NaiveBayesBasic()
        self.spam_similarities = {}
        self.ham_similarities = {}

    def get_embedding(self, sentences):
        #get w2v model
        # vector_length = 100
        # return WordEmbeddingsKeyedVectors(vector_length)
        return Word2Vec(sentences, window=5, workers=4, min_count=1)

    def calculate_centroid(self, embeddings, document):
        document_embeddings = [embeddings.wv[word] for word in [line for line in document]]
        # for line in document:
        #     wv = embedding.wv
        #     enumerable = wv.index_to_key
        #     # [wv[word] for word in enumerate(enumerable)]
        print(document_embeddings)
        return np.mean(document_embeddings, axis=0)
        # return np.mean(embeddings, axis=0)

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
        ham_words = []
        spam_words = []
        # self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        # self.model = AutoModel.from_pretrained("bert-base-uncased")
        print("training hams...")
        for ham_file in HAM_FILES:
            file = open(ham_file, "r", encoding="ISO-8859-1")
            ham_words.extend([line.strip().split() for line in file])
            file.close()
            # embeddings = self.get_embedding(words)
            # ham_embeddings.append(embeddings)
        ham_embeddings = self.get_embedding(ham_words)
        print("training spams...")
        for spam_file in SPAM_FILES:
            file = open(spam_file, "r", encoding="ISO-8859-1")
            spam_words.extend([line.strip().split() for line in file])
            file.close()
            # embeddings = self.get_embedding(words)
            # spam_embeddings.append(embeddings)
        spam_embeddings = self.get_embedding(spam_words)
        self.spam_embeddings = spam_embeddings
        self.ham_embeddings = ham_embeddings
        self.nbb.train()
        print("done training!")
        # self.ham_total = float(ham_total)
        # self.ham_freq = ham_freq
        # self.spam_total = float(spam_total)
        # self.spam_freq = spam_freq


        # spam_words = list(spam_freq.keys())
        # ham_words = list(ham_freq.keys())
        batch_size = 1024

        # spam_embeddings = []
        # for i in range(0, len(spam_words), batch_size):
        #     batch = spam_words[i:i + batch_size]
        #     embeddings = self.get_embedding(batch)
        #     spam_embeddings.extend(embeddings)

        # ham_embeddings = []
        # for i in range(0, len(ham_words), batch_size):
        #     batch = ham_words[i:i + batch_size]
        #     embeddings = self.get_embedding(batch)
        #     ham_embeddings.extend(embeddings)

        #centroid calculation stuff
        # ham_centroids = []
        # spam_centroids = []
        # for ham_file in HAM_FILES:
        #     file = open(ham_file, "r", encoding="ISO-8859-1")
        #     ham_centroids.append(self.calculate_centroid(ham_embeddings, [line.strip().split() for line in file]))
        #     file.close()
        #     # embeddings = self.get_embedding(words)
        #     # ham_embeddings.append(embeddings)
        # for spam_file in SPAM_FILES:
        #     file = open(spam_file, "r", encoding="ISO-8859-1")
        #     spam_centroids.append(self.calculate_centroid(spam_embeddings, [line.strip().split() for line in file]))
        #     file.close()
        
        # self.spam_centroids = spam_centroids
        # self.ham_centroids = ham_centroids

        # self.spam_centroid = self.calculate_centroid(spam_embeddings)
        # self.ham_centroid = self.calculate_centroid(ham_embeddings)

    def get_max_similarity_words(self, word, embeddings, k):
        maxima = {}
        wv = embeddings.wv
        # take first column - i'm not sure why these embeddings are 2x100
        norm_new = np.linalg.norm(word[0, :])
        for embedding in enumerate(wv.index_to_key):
            cos = np.dot(word[0, :], wv[embedding][0, :])/(norm_new*np.linalg.norm(wv[embedding][0, :]))
            if len(maxima) < k or float(cos) > min(maxima, key=maxima.get)[0]:
                maxima[embedding] = cos
        return maxima.keys()

    def predict(self, filename):
        file = open(filename, "r", encoding="ISO-8859-1")
        new_embedding = self.get_embedding(file.read())
        file_words = []
        for line in file:
            file_words.extend(line.strip().split())
        file.close()

        # distance_to_spam = np.linalg.norm(new_embedding - self.spam_centroid)
        # distance_to_ham = np.linalg.norm(new_embedding - self.ham_centroid)
        for word in enumerate(new_embedding.wv.index_to_key):
            ham_max_words = None
            spam_max_words = None
            if word in self.spam_similarities:
                spam_max_words = self.spam_similarities[word]
            else:
                spam_max_words = self.get_max_similarity_words(new_embedding.wv[word], self.spam_embeddings, 5)
                self.spam_similarities[word] = spam_max_words
            if word in self.ham_similarities:
                ham_max_words = self.ham_similarities[word]
            else:
                ham_max_words = self.get_max_similarity_words(new_embedding.wv[word], self.ham_embeddings, 5)
                self.ham_similarities[word] = spam_max_words
            file_words.extend(spam_max_words)
            file_words.extend(ham_max_words)
        print("done padding!")
        return self.nbb.predict_wordlist(file_words)



if __name__ == "__main__":
    nb = EmbeddingBayes()
    nb.train()
    accuracy = nb.test()
    print(accuracy)