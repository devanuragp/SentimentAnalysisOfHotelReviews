######################## DO NOT RUN THIS FILE ########################

######################## import all required files ########################

import pandas as pd
import nltk
import random
import string
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

from textblob import Word
from textblob import TextBlob

from tkinter import *

######################## Loading data ########################

F_DIRECTORY = "C:/Users/Anurag/Hotel_sentiment_analysis_2/dataset/dataset_random.csv";

df = pd.read_csv(F_DIRECTORY)

positive_reviews = df["Positive_Review"]
negative_reviews = df["Negative_Review"]


######################## Preprocessing data ########################

for positive_review in positive_reviews:
    with open("positive_reviews.txt", "a") as fileObject:
        if positive_review != "No Positive":
            fileObject.write(positive_review.lower())
            fileObject.write("\n")


for negative_review in negative_reviews:
    with open("negative_reviews.txt", "a") as fileObject:
        if negative_review != "No Negative":
            fileObject.write(negative_review.lower())
            fileObject.write("\n")

#opening the text files
short_pos = open("positive_reviews.txt","r").read()
short_neg = open("negative_reviews.txt","r").read()

all_words = []
documents = []

# J is adjective , R is adverb , V is verb
allowed_word_types = ["J","R","V"]


for p in short_pos.split('\n'):
    documents.append( (p, "positive" ) )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


for p in short_neg.split('\n'):
    documents.append( (p, "negative" ) )
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())
    
    
#pickle documents
            
save_documents = open("pickled_files/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()

##documents_f = open("documents.pickle","rb")
##documents = pickle.load(documents_f)
##documents_f.close()
    
    
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]


#pickle word_features

save_word_features = open("pickled_files/word_features5k.pickle","wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()

##word_f = open("word_features5k.pickle","rb")
##word_features = pickle.load(word_f)
##word_f.close()

#cleaning text function

def clean_text(text):
    # do word tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    return(words)

# list to string function

def listToString(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s))



######################## building featuresets ########################

#finding features function

def find_features(document):
    words = clean_text(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

##feature_f = open("pickled_files/features.pickle","rb")
##featuresets = pickle.load(feature_f)
##feature_f.close()

#pickle featureset

save_features = open("pickled_files/features.pickle","wb")
pickle.dump(featuresets, save_features)
save_features.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:25000]
testing_set = featuresets[25000:]


print("Original values \n")
print("Sentence 1 : ",testing_set[0][1])
print("Sentence 2 : ",testing_set[1][1])
print("Sentence 3 : ",testing_set[2][1])
print("Sentence 4 : ",testing_set[3][1])
print("Sentence 5 : ",testing_set[4][1])
print("Sentence 6 : ",testing_set[5][1])
print("Sentence 7 : ",testing_set[6][1])
print("Sentence 8 : ",testing_set[7][1])
print("Sentence 9 : ",testing_set[8][1])
print("Sentence 10 : ",testing_set[9][1])
print("Sentence 11 : ",testing_set[10][1])
print("Sentence 12 : ",testing_set[11][1])
print("Sentence 13 : ",testing_set[12][1])
print("Sentence 14 : ",testing_set[13][1])
print("Sentence 15 : ",testing_set[14][1],"\n######################################")


######################## training classifier ########################


###load Original NaiveBayes_classifier
##classifier_f = open("pickled_files/naivebayes.pickle","rb")
##classifier = pickle.load(classifier_f)
##classifier_f.close()

###load MNB Classifier
##classifier_f = open("pickled_files/MNB_classifier.pickle","rb")
##MNB_classifier = pickle.load(classifier_f)
##classifier_f.close()

###load BernoulliNB Classifier
##classifier_f = open("pickled_files/BernoulliNB_classifier.pickle","rb")
##BernoulliNB_classifier = pickle.load(classifier_f)
##classifier_f.close()

###load LogisticRegression Classifier
##classifier_f = open("pickled_files/LogisticRegression_classifier.pickle","rb")
##LogisticRegression_classifier = pickle.load(classifier_f)
##classifier_f.close()

###load SGDClassifier Classifier
##classifier_f = open("pickled_files/SGDClassifier_classifier.pickle","rb")
##SGDClassifier_classifier = pickle.load(classifier_f)
##classifier_f.close()

###load SGDClassifier Classifier
##classifier_f = open("pickled_files/LinearSVC_classifier.pickle","rb")
##LinearSVC_classifier = pickle.load(classifier_f)
##classifier_f.close()


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


#Original NaiveBayes Classifier
classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original NaiveBayes_classifier accuracy percent", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

("Naive Bayes Classifier \n")
print("Sentence 1 : ",classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",classifier.classify(testing_set[14][0]),"\n######################################")

#pickle classifier
save_classifier = open("pickled_files/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


#MNB Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

print("Multinomial Naive Bayes Classifier \n")
print("Sentence 1 : ",MNB_classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",MNB_classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",MNB_classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",MNB_classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",MNB_classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",MNB_classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",MNB_classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",MNB_classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",MNB_classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",MNB_classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",MNB_classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",MNB_classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",MNB_classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",MNB_classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",MNB_classifier.classify(testing_set[14][0]),"\n######################################")

#pickle classifier
save_classifier = open("pickled_files/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()


#BernoulliNB Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

print("Bernoulli Naive Bayes Classifier \n")
print("Sentence 1 : ",BernoulliNB_classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",BernoulliNB_classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",BernoulliNB_classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",BernoulliNB_classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",BernoulliNB_classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",BernoulliNB_classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",BernoulliNB_classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",BernoulliNB_classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",BernoulliNB_classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",BernoulliNB_classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",BernoulliNB_classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",BernoulliNB_classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",BernoulliNB_classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",BernoulliNB_classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",BernoulliNB_classifier.classify(testing_set[14][0]),"\n######################################")

#pickle classifier
save_classifier = open("pickled_files/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()


#LogisticRegression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

print("Logistic Regression Classifier \n")
print("Sentence 1 : ",LogisticRegression_classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",LogisticRegression_classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",LogisticRegression_classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",LogisticRegression_classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",LogisticRegression_classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",LogisticRegression_classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",LogisticRegression_classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",LogisticRegression_classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",LogisticRegression_classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",LogisticRegression_classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",LogisticRegression_classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",LogisticRegression_classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",LogisticRegression_classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",LogisticRegression_classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",LogisticRegression_classifier.classify(testing_set[14][0]),"\n######################################")

#pickle classifier
save_classifier = open("pickled_files/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


#SGDClassifier Classifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

print("Stochastic Gradient Descent Classifier \n")
print("Sentence 1 : ",SGDClassifier_classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",SGDClassifier_classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",SGDClassifier_classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",SGDClassifier_classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",SGDClassifier_classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",SGDClassifier_classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",SGDClassifier_classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",SGDClassifier_classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",SGDClassifier_classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",SGDClassifier_classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",SGDClassifier_classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",SGDClassifier_classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",SGDClassifier_classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",SGDClassifier_classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",SGDClassifier_classifier.classify(testing_set[14][0]),"\n######################################")

#pickle classifier
save_classifier = open("pickled_files/SGDClassifier_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()


#LinearSVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

print("LinearSVC Classifier \n")
print("Sentence 1 : ",LinearSVC_classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",LinearSVC_classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",LinearSVC_classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",LinearSVC_classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",LinearSVC_classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",LinearSVC_classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",LinearSVC_classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",LinearSVC_classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",LinearSVC_classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",LinearSVC_classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",LinearSVC_classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",LinearSVC_classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",LinearSVC_classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",LinearSVC_classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",LinearSVC_classifier.classify(testing_set[14][0]),"\n######################################")

#pickle classifier
save_classifier = open("pickled_files/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


#voted Classifier
voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier)

print("voted_classifier accuracy percent", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Voted Classifier \n")
print("Sentence 1 : ",voted_classifier.classify(testing_set[0][0]))
print("Sentence 2 : ",voted_classifier.classify(testing_set[1][0]))
print("Sentence 3 : ",voted_classifier.classify(testing_set[2][0]))
print("Sentence 4 : ",voted_classifier.classify(testing_set[3][0]))
print("Sentence 5 : ",voted_classifier.classify(testing_set[4][0]))
print("Sentence 6 : ",voted_classifier.classify(testing_set[5][0]))
print("Sentence 7 : ",voted_classifier.classify(testing_set[6][0]))
print("Sentence 8 : ",voted_classifier.classify(testing_set[7][0]))
print("Sentence 9 : ",voted_classifier.classify(testing_set[8][0]))
print("Sentence 10 : ",voted_classifier.classify(testing_set[9][0]))
print("Sentence 11 : ",voted_classifier.classify(testing_set[10][0]))
print("Sentence 12 : ",voted_classifier.classify(testing_set[11][0]))
print("Sentence 13 : ",voted_classifier.classify(testing_set[12][0]))
print("Sentence 14 : ",voted_classifier.classify(testing_set[13][0]))
print("Sentence 15 : ",voted_classifier.classify(testing_set[14][0]))

######################## sent_analysis functions ########################

def sentiment_classification(text):
    features = find_features(text)
    return voted_classifier.classify(features)

def score_text(text):
    text_in = clean_text(text)
    text_out = listToString(text_in)
    blob = TextBlob(text_out)
    return float("{0:.2f}".format(((blob.sentiment.polarity+1)/2)*10))

def summary_text(text):
    noun = []
    summary = []
    text_in = clean_text(text)
    text_out = listToString(text_in)
    blob = TextBlob(text_out)
    for word,tag in blob.tags:
        if tag =="NN":
            noun.append(word.lemmatize())
    
    noun=nltk.FreqDist(noun)
    features=list(noun.keys())[:5]
    
    for item in features:
        word = Word(item)
        summary.append(word.pluralize())
    
    return summary

def sentiment_analysis(text):
    return  sentiment_classification(text),score_text(text),summary_text(text)

######################## testing sent_analysis ########################

sentiment = sentiment_analysis("Tom is  not good boy . He misbehaves which good boys dont do ")

for i in sentiment:
    print(i)

######################## applying model to dataset ########################

# read data
reviews_df = pd.read_csv(r"C:\Users\Anurag\BE_proj_aspect_sentiment\Hotel_Reviews_copy2.csv")

# append the positive and negative text reviews
reviews_df["review"] = reviews_df["Negative_Review"] + reviews_df["Positive_Review"]

# data description ( may only work on jupyter notebook )

reviews_df.columns

print ('Number of data points : ', reviews_df.shape[0], \
       '\nNumber of features:', reviews_df.shape[1])

reviews_df.Hotel_Name.describe()

reviews_df.review.describe()


# remove 'No Negative' or 'No Positive' from text
reviews_df["review"] = reviews_df["review"].apply(lambda x: x.replace("No Negative", "").replace("No Positive", ""))

# Taking a sample of complete dataset for futher analysis

sample_df=reviews_df.sample(100000)
sample_df.head(5)

sample_df.columns

# selecting only important columns and discarding all others 

sample_df = sample_df[["Review_Date","Hotel_Name","Positive_Review","Negative_Review","review","Reviewer_Score","Total_Number_of_Reviews_Reviewer_Has_Given"]]
sample_df.head(5)

# data description of new df( may only work on jupyter notebook )

print ('Number of data points : ', sample_df.shape[0], \
       '\nNumber of features:', sample_df.shape[1])

sample_df.Hotel_Name.describe()

sample_df.review.describe()

# using lambda function to apply function & getting sentiment for reviews
sample_df["sentiment"]=sample_df["review"].apply(lambda x:sentiment_classification(x))

# score text data
sample_df["sentiment_score"] = sample_df["review"].apply(lambda x: score_text(x))

# summary of positive text data
sample_df["pos_summary"] = sample_df["Positive_Review"].apply(lambda x:summary_text(x))

# summary of negative text data
sample_df["neg_summary"] = sample_df["Negative_Review"].apply(lambda x:summary_text(x))

# saving new column with other columns to csv file
reviews_df.to_csv(r"C:\Users\Anurag\BE_proj_aspect_sentiment\Hotel_Reviews_copy4.csv")


######################## GUI for sent analyser ########################


def main():
    root = Tk()
    root.title("Sentiment Analyser")
    root.configure(bg='#FFFFFF')
    return root


def runprog(root):
    
    # Creating frame for label

    framelabel = Frame(root , bg='#FFFFFF')
    framelabel.pack(expand=YES, fill=BOTH, padx=16, pady=24)

    # Creating a label Widget

    myLabel = Label(framelabel, text = "Welcome to Hotel Reviews Sentiment analyser" , font =('Helvetica',16,'bold'),bg='#FFFFFF',anchor = CENTER)
    myLabel.pack(expand=YES, fill=BOTH, pady=24)
    myLabel2 = Label(framelabel, text = "This Sentiment analyser will classify sentiment into positive and negative, generate sentiment score and summary " , font =('Helvetica',10),bg='#FFFFFF',anchor = NW)
    myLabel2.pack(expand=YES, fill=BOTH)
    myLabel3 = Label(framelabel, text = "Enter your review in the given text box for getting the output " , font =('Helvetica',10),bg='#FFFFFF',anchor = NW)
    myLabel3.pack(expand=YES, fill=BOTH)


    # Creating a Text Widget

    TextArea = Text(root)
    ScrollBar = Scrollbar(root)
    ScrollBar.config(command=TextArea.yview)
    TextArea.config(yscrollcommand=ScrollBar.set, height = 10, width = 50, relief = GROOVE , bd = 5 )
    ScrollBar.pack(side=RIGHT, fill=Y)
    TextArea.pack(expand=YES, fill=BOTH, padx=16, pady=8)



    # Creating a frame for buttons

    frame = Frame(root , bg='#FFFFFF')
    frame.pack(padx=16, pady=16)

    # Creating frame for label

    framelabel1 = Frame(root , bg='#FFFFFF')
    framelabel1.pack(expand=YES, fill=BOTH, padx=16, pady=8)

    # Function to convert list to string

    def listToString(s):  
        # initialize an empty string 
        str1 = " " 
        # return string   
        return (str1.join(s)) 
        

    # Creating fuctions for buttons

    def submit():
        text=TextArea.get('1.0',END)
        #Sentiment classification
        sentiment = sentiment_classification(text)
        label1 = "Sentiment : "+sentiment
        # Creating a Label Widget
        myLabel= Label(framelabel1 , text =label1,justify = LEFT,anchor=NW , bg='#FFFFFF', font =('Helvetica',10))
        myLabel.pack(expand=YES, fill=BOTH,pady=3)
        #Sentiment score
        score = str(score_text(text))
        label2 = "Sentiment Score : "+score
        # Creating a Label Widget
        myLabel2= Label (framelabel1 , text =label2,justify = LEFT,anchor=NW, bg='#FFFFFF', font =('Helvetica',10))
        myLabel2.pack(expand=YES, fill=BOTH,pady=3)
        #Sentiment summary
        summary_list=summary_text(text)
        summary=listToString(summary_list)
        label3= "This review is about : "+summary
        # Creating a Label Widget
        myLabel3= Label (framelabel1 , text =label3,justify = LEFT,anchor=NW, bg='#FFFFFF', font =('Helvetica',10))
        myLabel3.pack(expand=YES, fill=BOTH,pady=3)

    #Creating exit button
        
    def exits():
        root.destroy()

    #Creating clear button
    def clear():
        TextArea.delete('1.0',END)

    def new():
        root.destroy()
        bone=main()
        runprog(bone)

    # Creating a Button Widget

    myButton3= Button(frame,text=" New ", padx=15, pady=2, command=new, relief = FLAT , bd = 5 )
    myButton3.config(bg='#FFFACD',activebackground='#FFFF99')
    myButton3.pack(side = LEFT, padx = 32)

    myButton= Button(frame,text=" Submit ", padx=15, pady=2, command=submit, relief = FLAT , bd = 5 )
    myButton.config( bg='#E5FFCC',activebackground='#CCFF99')
    myButton.pack(side = LEFT, padx = 32)

    myButton2= Button(frame,text=" Clear ", padx=15, pady=2, command=clear ,relief = FLAT , bd = 5 )
    myButton2.config( bd = 5, bg='#D3D3D3',activebackground='#C0C0C0')
    myButton2.pack(side = LEFT, padx = 32)

    myButton4= Button(frame,text=" Exit ", padx=15, pady=2, command=exits, relief = FLAT , bd = 5 )
    myButton4.config( bg='#FFCCCC',activebackground='#FF9999')
    myButton4.pack(side = LEFT, padx = 32)


#main

root = main()
runprog(root)
root.mainloop()
