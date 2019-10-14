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
# Getting information from text.txt inside the files folder. Here the documents with the assigment text is.
f = codecs.open("files/text.txt", "r", "utf-8")

# Part 1.2
# By splitting with \n, we can retirve a array with each paragraph
book = f.read()
documents = book.split("\n\n")
paragraphDocuments = documents

# Part 1.3
# Here we remove all the paragraph with the word gutenberg inside it

for i in range(len(documents)-1, 0, -1):
    if "gutenberg" in documents[i] or "Gutenberg" in documents[i]or "GUTENBERG" in documents[i]:
        documents.pop(i)

# Part 1.4
# The paragraph is split into words, creating Arrays of word for each paragraph.
# Making the documents with arrays of arrays
paragraphArrays = []
for paragraph in documents:
    paragraphArrays.append(paragraph.split())
documents = paragraphArrays


# Part 1.5
# For each word in this paragraphArrays, the word get lowercased and removed, if it contains it, the punctation.
# Everything saved in douments
for paragraphIndex in range(len(documents)):
    for wordIndex in range(len(documents[paragraphIndex])):
        documents[paragraphIndex][wordIndex] = documents[paragraphIndex][wordIndex].translate(str.maketrans('', '', string.punctuation)).lower()


# Part 1.6
# Each word get converted to its stemword
# Example ["Have","Has", "Having"] => ["Ha","Ha","Ha"]
stemmer = PorterStemmer()
for paragraphIndex in range(len(documents)):
    for wordIndex in range(len(documents[paragraphIndex])):
        documents[paragraphIndex][wordIndex] = stemmer.stem(documents[paragraphIndex][wordIndex])


# Part 1.7
# Here the count of frequencies get recorded on a FreqDist() object. 
freq = FreqDist()
for paragraph in documents:
    for word in paragraph:
        freq[word] += 1


# Section 2
# Part 2.1
# The building of a dictionary is made by using the the array of the paragraphArrays, mapping each word to a integer.
dictionary = gensim.corpora.Dictionary(documents)

# Part 2.2
# The stop words are copied from https://www.textfixer.com/tutorials/common-english-words.txt saved as an file in files called stopWords.txt
# Here the words are splitted to an arrays of each word by ","
with open("files/stopWords.txt", "r") as f:
    for line in f:
        stopWords = line
stopWords = stopWords.split(",")

# Here each stopword get checked if it exist in the dictonary getting the ID of the token if it does.
# This is then used to filter out all the stopword removing the its value from future calulationes 
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stopWords
    if stopword in dictionary.token2id
]
dictionary.filter_tokens(stop_ids)

#Now the dictonary will not return values to stopword. Its stop word free.
dictionary.compactify()

# Part 2.3
# The corpus is made by converting each array of paragraph in document to bow values. 
# Now the corpuse contains all the vectore values 
corpus = [dictionary.doc2bow(paragraph) for paragraph in documents]

# Section 3

# Part 3.1
# By using the corpus, a TF-IDF model can be generated
tfidf_model = gensim.models.TfidfModel(corpus)


# Part 3.2
# Each elemnet in the corpus  gets it TF-IDF value and word index.
tfidf_corpus = tfidf_model[corpus]
print(tfidf_corpus)

# Part 3.3
# Creating a similarity matrix to future calculate similarities between query and paragraphs
TF_IDF_MatrixS = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# Part 3.4
# Creating the LSI model using the corupus TF-IDF values
lsi_model = gensim.models.LsiModel(
    tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[corpus]
lsi_MatrixSim = gensim.similarities.MatrixSimilarity(lsi_corpus)

# Part 3.5
#Printing the 3 first parapgraphs most important topics ( highest values )
print("------Relevant top 3-----")
for i in lsi_model.show_topics(3):
    print(i)
print("-------------------------")

# Section 4
# Part 4.1

# preprossesing prosses the query into lowercaser array with stem word from the original query
def preprocessing(q):
    stemmer = PorterStemmer()
    q = q.lower().split()
    for wordIndex in range(len(q)):
        q[wordIndex] = q[wordIndex].translate(str.maketrans('', '', string.punctuation))
        q[wordIndex] = stemmer.stem(q[wordIndex].lower())
    return q


query = preprocessing("What is the function of money?")

#The query is then tranformed to a BOW element
query = dictionary.doc2bow(query)

# Part 4.2

# -------------------------------TEST CODE -------------------------------
# Testing if results are correct as given in the assigment
# Result should be : (tax: 0.26, econom: 0.82, influenc: 0.52)
queryPart4_2 = dictionary.doc2bow(
    preprocessing("How taxes influence Economics?"))

print("\n Testing values 'How taxes influence Economics?'")
answer = "( "
queryPart4_2 = tfidf_model[queryPart4_2]
print(queryPart4_2)
for tokenValue in queryPart4_2:
    answer += dictionary[(tokenValue[0])]+": "+str(tokenValue[1])+", "
print(answer[:-2]+" )\n")
# -------------------------------TEST CODE -------------------------------


# Same prosses on the wanted query already prossesed in part 4.1
print('q = "What is the function of money?"')
query = tfidf_model[query]
print(str(query)+"\n")



# Part 4.3
# Funtion for sorting funtion. Later used for sortin tupplets
def getKey(item):
    return item[1]

#For later printing only the first lines if longer than that.
def firstFiveLines(text, nr):
    text = text.split("\n")
    print("Relevant nr"+str(nr))
    for i in range(5):
        try:
            print(text[i])
        except:
            break
    print("\n")

# Find th similarities between the query and the paragraphs 
relevante = TF_IDF_MatrixS[query]

# Maybe not the best way, but iterate trow all elements to create a tuplet as (paragraphNumber, TF-IDF weight)
rel = []
for relevanteIndex in range(len(relevante)):
    rel.append((relevanteIndex, relevante[relevanteIndex]))

# Sorting the tuplet with index 1 (second element) and reversing 
rel = sorted(rel, key=getKey, reverse=True)

# Here the top 3 most relevant paragraphs 
print("Query: " + "What is the function of money?")
firstFiveLines(paragraphDocuments[rel[0][0]], 1)
firstFiveLines(paragraphDocuments[rel[1][0]], 2)
firstFiveLines(paragraphDocuments[rel[2][0]], 3)

# Part 4.4


lsiQuery = lsi_model[query]
topTopics = sorted(lsiQuery, key=lambda kv: - abs(kv[1]))[:3]
print("Top 3 topics with lsi topics weights:")
print(topTopics)
topics = lsi_model.show_topics(100)
for topic in topTopics:
    print("\n")
    print("LSI topic", topic[0], ":")
    print(topics[topic[0]])


# find the 3 topics most relevant paragraphs according to LSI model:
doc2similarity = enumerate(lsi_MatrixSim.get_similarities(lsiQuery))
print("\n")
sortedParagraphs2 = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]

print("------Relevant top 3-----")
for i in lsi_model.show_topics(3):
    print(i)
print("-------------------------")