from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import re
import string
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import NaiveBayes
from pyspark.sql.types import ArrayType, StringType
import numpy as np
import pandas as pd
from pyspark.sql.types import DoubleType, IntegerType
from handyspark import *
from sklearn import metrics

import pyspark.sql.functions as f

# NLTK imports
import nltk
nltk.download('punkt')
# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('averaged_perceptron_tagger')


st.write("Creating a spark session")
spark = SparkSession.builder.appName('TwitterSentiment').getOrCreate()
dataset = spark.read.csv("/content/US_Airline_Tweets.csv",inferSchema=True,header=True)
hdf = dataset.toHandy()

st.write("Preprocessing the train data")
# 1. Data preprocessing (PASTE YOUR ENTIRE DATA PREPROCESSING CODE FROM ABOVE)




hdf = hdf.dropna(subset=['text'])

# pdf = hdf.notHandy()
pdf = hdf.select("*", f.lower("text"))


from pyspark.sql.types import DoubleType, IntegerType

pdf = pdf.withColumn("tweet_id",pdf.tweet_id.cast('integer'))
pdf = pdf.withColumn("airline_sentiment_confidence",pdf.airline_sentiment_confidence.cast('double'))
pdf = pdf.withColumn("negativereason_confidence",pdf.negativereason_confidence.cast('double'))
pdf = pdf.withColumn("retweet_count",pdf.retweet_count.cast('integer'))


#UserlinkUDF Func def
#Start
def remove_usernames_links(tweet):
    tweet = re.sub('@[^\s]+','',tweet)
    tweet = re.sub('http[^\s]+','',tweet)
    tweet = re.sub('#[^\s]+','',tweet)

    return tweet

remove_usernames_linksUDF = udf(lambda z: remove_usernames_links(z), StringType())
spark.udf.register("remove_usernames_linksUDF", remove_usernames_linksUDF)
#End

for col in pdf.columns[1:]:
    if col != 'airline_sentiment':
        # print(col)
        # pdf = pdf.na.fill(value=0,subset=[col])
        pdf = pdf.na.fill(value=0,subset=[col])


ppdf = pdf.withColumn('text',remove_usernames_linksUDF('text'))

#Word_udf def
#Start
from nltk.tokenize import word_tokenize

word_udf = udf(lambda x: word_tokenize(x), ArrayType(StringType()))
#End
ppdfNew = ppdf.withColumn("text", word_udf("text"))

#Punctuation Def
#Start
stop_words = set(stopwords.words('english'))

punctuation = string.punctuation

punct_udf = udf(lambda x: [w for w in x if not w.lower() in punctuation if not w.lower() in stop_words])
cdf = ppdfNew.withColumn("text", punct_udf("text"))
#End

cdf = ppdfNew.withColumn("text", punct_udf("text"))

#lemmatize def
#Start
from nltk.stem import WordNetLemmatizer #should be included
nltk.download('averaged_perceptron_tagger') #should be included

def get_wordnet_pos(treebank_tag):
    """
    return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v) 
        """
    if treebank_tag.startswith('J'):
            return 'a'
    elif treebank_tag.startswith('V'):
            return 'v'
    elif treebank_tag.startswith('N'):
            return 'n'
    elif treebank_tag.startswith('R'):
            return 'r'
    else:
    # As default pos in lemmatization is Noun
        return 'n'


def lemmatize1(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    #text = data_str.split()
    tagged_words = nltk.pos_tag(data_str)
    for word in tagged_words:
        lemma = lmtzr.lemmatize(word[0], get_wordnet_pos(word[1]))
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

sparkLemmer = udf(lambda x: lemmatize1(x), StringType())
#End
cdfNew = cdf.withColumn("text", sparkLemmer("text"))


st.write("Ongoing feature extraction!!")
# 2. Feature Extraction (PASTE YOUR ENTIRE FEATURE EXTRACTION CODE FROM ABOVE)

def calculateTweet(strText,flag):
    if flag == 'L':
        return len(strText)
    if flag == 'H':
        return len(re.findall("#", strText))
    if flag == 'A':
        return len(re.findall("@", strText))

tweetCalcL = udf(lambda x: calculateTweet(x,'L'), IntegerType())
tweetCalcH = udf(lambda x: calculateTweet(x,'H'), IntegerType())
tweetCalcA = udf(lambda x: calculateTweet(x,'A'), IntegerType())

DF1 = cdfNew.withColumn('Tweet_Length',tweetCalcL('text'))
DF2 = DF1.withColumn('Tweet_HashLen',tweetCalcH('text'))
DF3 = DF2.withColumn('Tweet_ATRLen',tweetCalcA('text'))

#encoderUDF Def
#Start
def LabelEncoder(x):
    if x == 'positive':
        return 0
    elif x == 'negative':
        return 1
    return 2

encoderUDF = udf(lambda x: LabelEncoder(x), IntegerType())
#End
DF4 = DF3.withColumn('Airline_SentimentN',encoderUDF('airline_sentiment'))



st.write("Training the model")
# 3. Training the model (PASTE YOUR MODEL TRAINING CODE FROM ABOVE)


assembler = VectorAssembler(
    inputCols=["retweet_count", "airline_sentiment_confidence", "negativereason_confidence", "Tweet_Length", "Tweet_HashLen", "Tweet_ATRLen"],
    outputCol="Features")

output = assembler.transform(DF4)

final_data = output.select("Features",'Airline_SentimentN')
train_data,test_data = final_data.randomSplit([0.7,0.3])

nb = NaiveBayes(featuresCol='Features', labelCol='Airline_SentimentN')
# Fit the model with train data
model = nb.fit(train_data)
predict = model.transform(test_data)

y_true = np.array(test_data.select('Airline_SentimentN').collect())
y_pred = np.array(predict.select('prediction').collect())
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_true,y_pred)*100)

st.write("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_true,y_pred)*100)
