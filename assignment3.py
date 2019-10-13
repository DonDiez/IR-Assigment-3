import codecs
import random
import string
from six import iteritems
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
import gensim

# Section 1

#   Part 1.0
random.seed(123)

# Part 1.1
f = codecs.open("files/text.txt", "r", "utf-8")

# Part 1.2
book = f.read()
documents = book.split("\n\n")


print(len(documents))
# Part 1.3
#print(book.split("\n\n")[2019])

for i in range(len(documents)-1,0,-1):
    if "gutenberg" in documents[i] or "Gutenberg" in documents[i]or "GUTENBERG" in documents[i]:
        documents.pop(i)       

# Part 1.4
paragraphArray = []
for paragraph in documents:
    paragraphArray.append(paragraph.split())
documents = paragraphArray


#Part 1.5
for paragraphIndex in range(len(documents)):
    for wordIndex in range(len(documents[paragraphIndex])):
        documents[paragraphIndex][wordIndex] = documents[paragraphIndex][wordIndex].translate(str.maketrans('', '', string.punctuation)).lower()


#Part 1.6
stemmer = PorterStemmer()
for paragraphIndex in range(len(documents)):
    for wordIndex in range(len(documents[paragraphIndex])):
        documents[paragraphIndex][wordIndex] = stemmer.stem(documents[paragraphIndex][wordIndex])


#Part 1.7
freq= FreqDist()
for paragraph in documents:
    for word in paragraph:
        freq[word]+=1

f
#Section 2

#Part 2.1
dictionary = gensim.corpora.Dictionary(documents)
print(dictionary)

#Part 2.2
with open("files/stopWords.txt", "r") as f:
    for line in f:
        stopWords = line 
stopWords = stopWords.split(",")

stopword = "that"
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stopWords
    if stopword in dictionary.token2id
]
dictionary.filter_tokens(stop_ids) 
dictionary.compactify() 

#Part 2.3
corpus = [dictionary.doc2bow(paragraph) for paragraph in documents]

#Section 3

#Part 3.1
tfidf_model = gensim.models.TfidfModel(corpus)

#Part 3.2
tfidf_corpus = tfidf_model[corpus]

#Part 3.3
MatrixSim = gensim.similarities.MatrixSimilarity(tfidf_corpus)

#Part 3.4
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[corpus]
lsi_MatrixSim = gensim.similarities.MatrixSimilarity(lsi_corpus)

#Part 3.5 
for i in lsi_model.show_topics(3):
    print(i)

# Section 4
