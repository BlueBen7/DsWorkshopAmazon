from __future__ import print_function
import random
import glob
import pandas as pd
import csv
import json
import datetime
from contextlib import contextmanager
from os.path import getsize, basename
import requests
import re
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import gzip
import dill as pickle
import textstat
import math
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_footer
from sklearn.datasets.twenty_newsgroups import strip_newsgroup_quoting
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from textblob import TextBlob
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import nltk
import string
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import minmax_scale
from spellchecker import SpellChecker
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
import gc
from clint.textui import progress
from tqdm import tqdm_notebook as tqdm
import requests
import time
from sklearn.preprocessing import PowerTransformer
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import ParameterGrid
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

spell = SpellChecker()

def pickleRead(root, pickle_type, filename):
    data_f = open(root + pickle_type + ("/%s.pickle" % filename), "rb")
    data = pickle.load(data_f)
    data_f.close()
    return data

def pickleWrite(root, pickle_type, variable, filename):
    saveData = open(root + pickle_type + ("/%s.pickle" % filename), "wb")
    pickle.dump(variable, saveData)
    saveData.close()

def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"

  session = requests.Session()
  headers = {'Range':'bytes=0-'}
  response = session.get(URL,headers=headers, params = { 'id' : id }, stream = True)
  token = get_confirm_token(response)

  if token:
      params = { 'id' : id, 'confirm' : token }
      response = session.get(URL,headers=headers, params = params, stream = True)

  save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    rng = response.headers.get('Content-Range')
    cont_leng=int(rng.partition('/')[-1])
    pbar = tqdm(
        total=cont_leng, 
        unit='B', unit_scale=True, desc=destination)
    #assert 0 == 1
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(CHUNK_SIZE)

#@title Infra Definitions
@contextmanager
def pbopen(filename, mode='r'):
    total = getsize(filename)
    pb = tqdm(total=total, unit="B", unit_scale=True,
              desc=basename(filename), miniters=1,
              ncols=80, ascii=True)

    def wrapped_line_iterator(fd):
        processed_bytes = 0
        for line in fd:
            processed_bytes += len(line)
            # update progress every MB.
            if processed_bytes >= 1024 * 1024:
                pb.update(processed_bytes)
                processed_bytes = 0

            yield line

        # finally
        pb.update(processed_bytes)
        pb.close()

    with open(filename, mode) as fd:
        yield wrapped_line_iterator(fd)
        
#decompress input folder to output folder
def ungzip(source_dir, dest_dir):
  for src_name in glob.glob(os.path.join(source_dir, '*.gz')):
      base = os.path.basename(src_name)
      dest_name = os.path.join(dest_dir, base[:-3])
      with gzip.open(src_name, 'rb') as infile:
          with open(dest_name, 'wb') as outfile:
              for line in infile:
                  outfile.write(line)

#Add isHelpful column
def add_meta_features(dataset):
  dataset['helpful'] = ApplyHelpfulnessVector(dataset) 
  
def ApplyHelpfulnessVector(reviews):
  return reviews.apply(lambda row: isHelpful(row['total_votes'], row['helpful_votes']), axis=1)

def isHelpful(total_votes, helpful_votes):
  if   (total_votes >= 5 and helpful_votes >= (total_votes * 0.6)):
    return 1
  elif (total_votes >= 5 and helpful_votes <  (total_votes * 0.6)):
    return -1
  else:
    return 0

#@markdown ###Review Filters
def isBiggerDate(date1, date2):
  year1,month1,day1 = [int(x) for x in date1.split('-')]
  year2,month2,day2 = [int(x) for x in date2.split('-')]
  ydiff,mdiff,ddiff = year1-year2,month1-month2,day1-day2
  if (ydiff>0): return True;
  elif (ydiff<0): return False;
  if (mdiff>0): return True;
  elif (mdiff<0): return False;
  if (ddiff>0): return True;
  elif (ddiff<0): return False;
  return False

def get_set_info(subset):
  return { 'helpful': np.sum(subset['helpful']== 1),
           'neutral': np.sum(subset['helpful']== 0),
           'unhelpful': np.sum(subset['helpful']== -1) }

def print_set_info(name, set_info):
  print(name, 'set size:', sum(set_info.values()))
  print('* Helpful:', set_info['helpful'])
  print('* Neutral:', set_info['neutral'])
  print('* Unhelpful:', set_info['unhelpful'])
  print('** Total helpful + unhelpful:', set_info['helpful']+set_info['unhelpful'],'\n')

#Features
def dateToInt(dateStr):
  year, month, day = dateStr.split('-')
  return 365*int(year)+30*int(month)+int(day)

def numWords(text):
  return len(re.findall("[a-zA-Z_]+", text))

def numSpellingMistakes(text):
  numMistakes = float(len(spell.unknown(text.split())))
  textLen = float(len(text))
  if (textLen<=0): return 0.0
  return numMistakes/textLen

def add_sentiment_statistics(df,column_title):
  split_sentences = []
  for index, row in df.iterrows():
    split_sentences.append(tokenize.sent_tokenize(row[column_title]))
    
  split_sentences = [[TextBlob(sentence).sentiment for sentence in review] for review in split_sentences]

  split_subjectivity = [[a.subjectivity for a in senten] for senten in split_sentences] 
  subjectivity_avg = [np.mean(review) for review in split_subjectivity]
  subjectivity_std = [np.std(review) for review in split_subjectivity]

  split_polarity = [[a.polarity for a in senten] for senten in split_sentences]
  polarity_avg = [np.mean(review) for review in split_polarity]
  polarity_std = [np.std(review) for review in split_polarity]

  df['subjectivity_avg'] = subjectivity_avg
  df['subjectivity_std'] = subjectivity_std
  df['polarity_avg'] = polarity_avg
  df['polarity_std'] = polarity_std

#A dictionary used to calculate a word's original form.
original_forms = {}

#Objects that are needed by best_tokenizer and can be computed once.
english_stopwords = stopwords.words('english') + ['i\'ve']
tokenizer_regex = RegexpTokenizer('^[a-zA-Z\']+$')
ess = SnowballStemmer('english', ignore_stopwords=True)

def best_tokenizer(sentence):
  tokens = tokenizer_regex.tokenize(sentence.replace(' ','\n'))
  tokens = [token.lower() for token in tokens]
  tokens = [token for token in tokens if token not in english_stopwords]

  stemmed = []
  for token in tokens:
    word_stem = ess.stem(token)
    stemmed.append(word_stem)
    
    # Compute original forms.
    if word_stem not in original_forms:
      original_forms[word_stem] = {token : 1}
    else:
      if token not in original_forms[word_stem]:
        original_forms[word_stem][token] = 1
      else:
        original_forms[word_stem][token] += 1

  return stemmed

#########

quick_conf_mat = None

def print_scores(Y_test, y):
  global quick_conf_mat
  
  score_test_accuracy = metrics.accuracy_score(Y_test, y)
  score_test_f1 = metrics.f1_score(Y_test, y)
  score_test_precision = metrics.precision_score(Y_test, y)
  score_test_recall = metrics.recall_score(Y_test, y)
  quick_conf_mat = metrics.confusion_matrix(Y_test, y)

  false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y)
  score_auc = auc(false_positive_rate, true_positive_rate)

  print("Model accuracy score: %.2f%%" % (100 * score_test_accuracy))
  print("Model f1 score: %.2f%%" % (100 * score_test_f1))
  print("Model precision: %.2f%%" % (100 * score_test_precision))
  print("Model recall: %.2f%%" % (100 * score_test_recall))
  print("Model AUC: %.2f%%" % (100 * score_auc))


def plot_hists(dataset, nrows, ncols, title=""):
  original_cols = list(dataset.columns)
  plt.figure(figsize=(12, 16))
  for i in range(len(original_cols)):
    colname = original_cols[i]
    plt.subplot(nrows, ncols, i+1)
    plt.title(colname)
    dataset[colname].plot.hist(bins=200)

  plt.suptitle(title)
  plt.tight_layout()
  plt.subplots_adjust(top=0.95)
  plt.show()

#get names util
class GetNames(BaseEstimator, TransformerMixin):
  names = []

  def fit(self, x, y=None):
      return self

  def transform(self, var):
      self.names = list(var.columns)
      return var

#get pipeline stages in nested pipelines
def getLeafPipelines(toupleList):
  numPipelines = 0
  newToupleList = []
  for touple in toupleList:
    itemName = touple[0]
    item = touple[1]
    if (type(item)==Pipeline):
      newToupleList.append(touple)
      numPipelines += 1
    elif (type(item)==FeatureUnion):
      for newTouple in item.transformer_list:
        newToupleList.append((itemName+"-"+newTouple[0],newTouple[1]))
  if (numPipelines==len(toupleList)):
    return newToupleList
  return getLeafPipelines(newToupleList)


def get_feature_scores(main_pipeline, allPipelines):
  # Search for feature names and scores.
  all_features = []
  for pipeTouple in allPipelines:
    smallPipeName = pipeTouple[0]
    smallPipe = pipeTouple[1]
    pipeSteps = list(smallPipe.named_steps.keys())
    if 'tfidf' in pipeSteps and 'kbest' in pipeSteps:
      feature_names = smallPipe.named_steps['tfidf'].get_feature_names()
      selected_k_indexes = smallPipe.named_steps['kbest'].get_support(indices=True)
      best_features_names = [feature_names[x] for x in selected_k_indexes]
      all_features += [smallPipeName+"="+x for x in best_features_names]
    elif 'vect' in pipeSteps:
      feature_names = smallPipe.named_steps['vect'].get_feature_names()
      all_features += feature_names
    elif 'names' in pipeSteps:
      feature_names = smallPipe.named_steps['names'].names
      all_features += feature_names
    else:
      print("Error: tfidf+kbest/vect not found in pipeline.")

  final_kbest_obj = main_pipeline.named_steps['kbest']

  selected_k_indexes = final_kbest_obj.get_support(indices=True)
  best_scores = [final_kbest_obj.scores_[x] for x in selected_k_indexes]
  sorted_best_features = sorted(zip(best_scores, all_features), reverse=True)

  # Remove NaNs. This happens if transformer_weight=0 for some feature. 
  sorted_best_features = [x for x in sorted_best_features if not math.isnan(x[0])]
  return sorted_best_features


def print_features_and_scores(sorted_best_features, maximum_to_print=10):
  for i in range(min(maximum_to_print, len(sorted_best_features))):
    x = sorted_best_features[i]
    print("%d. %s (score: %.2f)" % (i+1, x[1], x[0]))

def get_best_textual_features(pipeline_textual,X_train,Y_train,feature_union):
  pipeline_textual.fit(X_train, Y_train)
  leaf_pipelines = getLeafPipelines([('txt', feature_union)])
  return get_feature_scores(pipeline_textual, leaf_pipelines)
  
def get_best_numerical_features(pipeline_numerical,X_train,Y_train):
  pipeline_numerical.fit(X_train, Y_train)
  leaf_pipelines = getLeafPipelines([('num', pipeline_numerical)])
  return get_feature_scores(pipeline_numerical, leaf_pipelines)

def __get_original_form_word(token):
  return max(original_forms[token], key=original_forms[token].get)

def get_original_form(token):
  return ' '.join([__get_original_form_word(t) for t in token.split()])

def clean_and_get_original_form(token):
  token = re.sub(r"^[^=]*=", "", token)
  return get_original_form(token)

def word_cloud(word_to_float_dict):
  # Generate a word cloud image
  wordcloud = WordCloud(background_color="white", max_words=100,
                       width=2800, height=1400).generate_from_frequencies(word_to_float_dict)

  # Display the generated image
  plt.figure(figsize = (18, 12))
  plt.imshow(wordcloud, interpolation='bilinear')
  plt.axis("off")
  plt.show()

def show_conf_matrix(cm):
  fig, ax = plot_confusion_matrix(conf_mat=cm, colorbar=True)

  labels = ['Unhelpful', 'Helpful']
  tick_marks = np.arange(len(labels))
  plt.xticks(tick_marks, labels, rotation=45)
  plt.yticks(tick_marks, labels)
  plt.suptitle("Confusion Matrix - Number of Predictions")

  plt.xlabel("Predicted Values")
  plt.ylabel("Actual Values")
  plt.show()

def plot_star_rating_distribution(df, plot_title):
  df_star_count = df.groupby("star_rating").size()
  row_count = df.shape[0]
  df_star_count_distribution = df_star_count.apply(lambda x: x/row_count)
  df_star_count_plot = df_star_count_distribution.plot.bar(rot=0, title=plot_title)
  return df_star_count_plot

FILE_DB_ID = 0
FILE_DB_EXISTANCE = 1

FILE_DB_ORIG = "file_db"
FILE_DB_NO_NEUTRALS = "file_db_no_neutrals"

init_file_db_original = {
                "Apparel.tsv" : ["15J2qWrfixNZtuKP_rythZcdZ7jMBgJma", False],
                "Watches.tsv" : ["15Uorj0owdJrEeM1-EQgBeIv_jFHJUAw_", False],
                "Beauty.tsv" : ["150TLBOBgaSJeDXQ1YGIRb3meKGDBei20", False],
                "Baby.tsv" : ["152OE8vEvSe050QfyThf4pZXNtjV2Hb49", False],
                "Automotive.tsv" : ["15CJFjDB0g3bmg4xRWu7M4RPTSNB0Wdhu", False],
                "Books_v1_00.tsv" : ["14wLw92wpbeg1o50MDyQKGz_o-kHxBY9n", False],
                "Digital_Software.tsv" : ["14arSG9tVjhQq78mIOfctl4xo-lV1R-Df", False],
                "Digital_Music_Purchase.tsv" : ["14fNXNqT6xrvGNBzyNuBsPEMJ3Nvpl5cK", False],
                "Digital_Ebook_Purchase1.tsv" : ["14lWd7SfLLF7x4b91HXxG_9N3A8Oq305W", False],
                "Camera.tsv" : ["14lf-vbGN6WrCjOg-cQHF-qfqixlin44h", False],
                "Books.tsv" : ["14nY9vnmgr4a4UySg1yxdPD2zJfmxvfUP", False],
                "Digital_Ebook_Purchase.tsv" : ["14uKr9RzhV4rBUbMwSW4mF_5JGzdoq0en", False],
                "Books_v1_01.tsv" : ["14vp1Jt8-Z512T3JQ0ipUY0o_xsEkGKvD", False],
                "Health_Personal_Care.tsv" : ["13nj0saFtD1ZwQ93hU0simTJZ3tI7Knx5", False],
                "Grocery.tsv" : ["13qlyWRZ0MuGZaRSakQ3GfPh3AQTHwHqK", False],
                "Gift_Card.tsv" : ["13xvYHdHOpjbdAP4K5MHdmg9zQn2nbXj0", False],
                "Furniture.tsv" : ["144Wts_J9_PrWCeiQCp6WP-ESsbEPrj53", False],
                "Electronics.tsv" : ["1464Te5hMOaU5ZgqPtsyaFTebjdIJleVZ", False],
                "Digital_Video_Download.tsv" : ["14Vi6UYmHsrO6ydvqXLkUfV_x1k7oLQ_T", False],
                "Mobile_Apps.tsv" : ["13AQ3G65NYfrunooRGHFdvXQMB6AsNTLy", False],
                "Major_Appliances.tsv" : ["13PH1aq_dhoEPGfTs5eMbypvaP9E_yCyW", False],
                "Luggage.tsv" : ["13Y7lwdOtvYdvkP7Qpjwf3RusIwBTnP1B", False],
                "Lawn_and_Garden.tsv" : ["13ZtVB_ORCxrt9WBT5E1m8d0TFgfoylMT", False],
                "Kitchen.tsv" : ["13ipSTsOu-OFkavz5z7hONJrpojd5gk2F", False],
                "Jewelry.tsv" : ["13kC_eK-UavQlMoqtMu31M6dIB_-wNfzS", False],
                "Home_Improvement.tsv" : ["13l4eBvnbF9qZjTJBZOKhXe5n2XUhyevP", False],
                "Home_Entertainment.tsv" : ["13lZkUkrzNoJDL1G8GhC41MEFpdw0vCe8", False],
                "Home.tsv" : ["13mokMehSe4-V3cMDDc9D5fgK--dO-9dh", False],
                "Personal_Care_Appliances.tsv" : ["12qT3fZN7yGdtwl2d3iu-45cZ9VphhuwN", False],
                "PC.tsv" : ["12qtRsF0nidSK8D8QGEbRCSFZt19VJY_t", False],
                "Outdoors.tsv" : ["12tt2ShYdeer3T3wwu4L-8dz2P09em8pG", False],
                "Office_Products.tsv" : ["1309gBj2VUQ_AcF9LTG398NpduxRV0TwY", False],
                "Musical_Instruments.tsv" : ["130xSLM06hQoScqe8DinMwY-GV98oDljm", False],
                "Music.tsv" : ["134aIeU5qeld2kax1ZiBvPjEh1e3EjgtY", False],
                "Mobile_Electronics.tsv" : ["136CCh7Ux2VRkykqg9JvmrsolCFwqDgwe", False],
                "Video.tsv" : ["12ZcuCAGoqW-sbMY10pWZxKPj5kwrgtaj", False],
                "Toys.tsv" : ["12_yv9wq-cr_y9RJBGR2gFrpc7Zmz3VdS", False],
                "Tools.tsv" : ["12fZijUQFu3Ce-pA5piOBUYOqXUikkyEx", False],
                "Sports.tsv" : ["12gm91nEibXS-GkFoQo9Q5uoA6GpatqPO", False],
                "Software.tsv" : ["12i932su0Ofda7kMtZcfT-BXpLo4tk1vO", False],
                "Shoes.tsv" : ["12iKvxbhyi2yNqETHs-CdI0z3uX4s8P4T", False],
                "Pet_Products.tsv" : ["12qKBaGhFv9hKti4PAwDqQVCTfVbkUvdE", False],
                "Wireless.tsv" : ["1-6_fpo5NKoqHPXt5_6NJwTFX3i0-qrEy", False],
                "Video_Games.tsv" : ["1-82C-cpVxRIh1RSg-o62TYQkjWpAB6y-", False],
                "Digital_Video_Games.tsv" : ["1-8LKqrsJIzE0A0_dq8fYVLTcAoVZwTLA", False],
                "Video_DVD.tsv" : ["12TTXWbRFjkEAbZI81c3zkGd8bxC0Jfkf", False],
               }

init_file_db_no_neutrals = {
              "Apparel.tsv" : ["1-4TmWinZ4tHN5L9RLisCb1P3nrZ-ypAH", False],
              "Automotive.tsv" : ["1-5WYYYcDzsbpevpTEbo5wi4ah-jNJAZ6", False],
              "Baby.tsv" : ["1-D3NiXVK4JPLCvhLAM_UKzl0wxlEN_Z-", False],
              "Beauty.tsv" : ["1-FjxxpbwQT1VS4XHHeirfmtvoDpVsT09", False],
              "Books.tsv" : ["1q4YTPc92TH61N84IptA5jmo9d6VG2nC7", False],
              "Books_v1_00.tsv" : ["1jaG36ny5Yxa3qrgd0ELaK9KZw3OWeI-g", False],
              "Books_v1_01.tsv" : ["1I01c0IHrGN3HiuIoSAmjWn888fkow_pL", False],
              "Camera.tsv" : ["1-0bOjYAlc6a-UdTOS9fKAFdd0SvtF9_b", False],
              "Digital_Ebook_Purchase.tsv" : ["1-EqJBx11bUWydi14ro1SMpyUTdVMpXV2", False],
              "Digital_Ebook_Purchase1.tsv" : ["1in9tdxTNi2sYsuKH6Ve95R_R2SKznDFr", False],
              "Digital_Music_Purchase.tsv" : ["1--STpm2s2ryN4AgS3xlgyg7aRWyCkcjj", False],
              "Digital_Software.tsv" : ["1-4E8wJmPIWr7umopiVzoo5krQElNWxg3", False],
              "Digital_Video_Download.tsv" : ["1-5RLlE1gq-KOMI2YJmmrshkODQZSVcu3", False],
              "Digital_Video_Games.tsv" : ["1-6Cxfg-i0mjfLwjw57g8E6CmYxjtMvgn", False],
              "Electronics.tsv" : ["1-EJE-2bHcBwJ8qe10WtliFUsDWJOO6k6", False],
              "Furniture.tsv" : ["1-F3VGgr6OmgQjwx5fF9mBxZMEEKCiNby", False],
              "Gift_Card.tsv" : ["1-QDp7q3MAlE3bH2bh9O7TNN9tAgDHXt0", False],
              "Grocery.tsv" : ["1-R0XjkTXsF0q_6hUz1jCJtIEaMqrMsqW", False],
              "Health_Personal_Care.tsv" : ["1-TdLr00rb33npHZSr6Ylz-QUFS0k9amP", False],
              "Home.tsv" : ["1-UjWOBS_RmoA6rdo_3WbIJjNzAI2xUY1", False],
              "Home_Entertainment.tsv" : ["1-XCM3NWvWPGqLiNyWO3Ew9XxBkFfllCH", False],
              "Home_Improvement.tsv" : ["1-j13y-cFZ0lbEWmmTo9PIGgocqd8p3Mu", False],
              "Jewelry.tsv" : ["1-m9KkUCNQj2NohMc9sEDfKha-TYrHdWu", False],
              "Kitchen.tsv" : ["1-xUeXkAYCs7uT3PskwQZustMXLElolcE", False],
              "Lawn_and_Garden.tsv" : ["1-yQtwhBoTrm3bvNQ3LYvCAYwVW7wVtjB", False],
              "Luggage.tsv" : ["103mrM6BggbXtkOD0w_HvJ4CrHXqXPtkD", False],
              "Major_Appliances.tsv" : ["107CuhJa2UJeLYwe-ZR4KSe8lHDNO0AWY", False],
              "Mobile_Apps.tsv" : ["10Dike2qTaSp8TW2KsmZgsFrhjyStTGd-", False],
              "Mobile_Electronics.tsv" : ["10RaCEohA48HO0eKgDJE1L8pvn6ovwdKt", False],
              "Music.tsv" : ["10_MRpd8YfA6GJ1s6ABam43UmjQrYuclO", False],
              "Musical_Instruments.tsv" : ["10qp5c9A0CjNFtwiRzUnLfHBDtVjLGRFw", False],
              "Office_Products.tsv" : ["10ry-Tlt96GJeDUpTXIKT0EkU0gaAMz3L", False],
              "Outdoors.tsv" : ["10wXs21GRai_cdvCuEn1t81mAkkGfaqF6", False],
              "PC.tsv" : ["1104aR_2BeMMEaZSSGU7X_KysOREKR5AF", False],
              "Personal_Care_Appliances.tsv" : ["110FE0Td1RmnXwRSiWh-KD4EnsE5xgSvQ", False],
              "Pet_Products.tsv" : ["110Lx6y9ZVAjV0K5msVpFKLFtCVmZCHlm", False],
              "Shoes.tsv" : ["111NsKKmd211BX1OZAqIiLL-kH7U1PkxG", False],
              "Software.tsv" : ["115I2QAXROLomQ03yaLWgcLmVqxeBmyXW", False],
              "Sports.tsv" : ["11CfJfGFLEbvlCOBwAw6PnTcOU9LV4rou", False],
              "Tools.tsv" : ["11F6Ui-8VoXY6OD2FMnWZwJa2SjBAtdLu", False],
              "Toys.tsv" : ["11ITxlXQH2TguO87vDAAo6V0T3WBzKNKE", False],
              "Video.tsv" : ["11JH6KMNA3efFcAPkF9-x49CBj7MPNXIo", False],
              "Video_DVD.tsv" : ["11Mtp3vyG32xO_5cpbJjLutF-jtGkAYaw", False],
              "Video_Games.tsv" : ["11SZVW1IIuhJFCVaLwnAwnDuvavGqUHXD", False],
              "Watches.tsv" : ["11SwuJa4ba8Ppa982IRq0-36ySez4j6Ib", False],
              "Wireless.tsv" : ["11U8z7PQXFIuKVXI0Tr7vl8SEM4lLEdTy", False],
               }

file_sizes_non_neutral_datasets = {
      'Apparel.tsv' : 230221,
      'Automotive.tsv' : 153099,
      'Baby.tsv' : 107685,
      'Beauty.tsv' : 349724,
      'Books.tsv' : 1418511,
      'Books_v1_00.tsv' : 630435,
      'Books_v1_01.tsv' : 1495443,
      'Camera.tsv' : 191541,
      'Digital_Ebook_Purchase.tsv' : 360498,
      'Digital_Ebook_Purchase1.tsv' : 374650,
      'Digital_Music_Purchase.tsv' : 36312,
      'Digital_Software.tsv' : 8610,
      'Digital_Video_Download.tsv' : 74872,
      'Digital_Video_Games.tsv' : 9575,
      'Electronics.tsv' : 211890,
      'Furniture.tsv' : 80027,
      'Gift_Card.tsv' : 882,
      'Grocery.tsv' : 158741,
      'Health_Personal_Care.tsv' : 491054,
      'Home.tsv' : 413825,
      'Home_Entertainment.tsv' : 90273,
      'Home_Improvement.tsv' : 183164,
      'Jewelry.tsv' : 64333,
      'Kitchen.tsv' : 396753,
      'Lawn_and_Garden.tsv' : 219379,
      'Luggage.tsv' : 28784,
      'Major_Appliances.tsv' : 18321,
      'Mobile_Apps.tsv' : 381737,
      'Mobile_Electronics.tsv' : 5976,
      'Music.tsv' : 741615,
      'Musical_Instruments.tsv' : 77019,
      'Office_Products.tsv' : 187336,
      'Outdoors.tsv' : 175754,
      'PC.tsv' : 348331,
      'Personal_Care_Appliances.tsv' : 11981,
      'Pet_Products.tsv' : 165006,
      'Shoes.tsv' : 161524,
      'Software.tsv' : 68214,
      'Sports.tsv' : 296677,
      'Tools.tsv' : 140003,
      'Toys.tsv' : 307859,
      'Video.tsv' : 100447,
      'Video_DVD.tsv' : 696093,
      'Video_Games.tsv' : 181240,
      'Watches.tsv' : 52999,
      'Wireless.tsv' : 288635
    }

file_sizes_original_datasets = {
        'Apparel.tsv': 5906333, 
        'Automotive.tsv': 3514942,
        'Baby.tsv': 1752932,
        'Beauty.tsv': 5115666,
        'Books.tsv': 3105520, 
        'Books_v1_00.tsv': 10319090, 
        'Books_v1_01.tsv': 6106719, 
        'Camera.tsv': 1801974,
        'Digital_Ebook_Purchase.tsv': 5101693,
        'Digital_Ebook_Purchase1.tsv': 12520722,
        'Digital_Music_Purchase.tsv': 1688884, 
        'Digital_Software.tsv': 102084,
        'Digital_Video_Download.tsv': 4057147,
        'Digital_Video_Games.tsv': 145431,
        'Electronics.tsv': 3093869,
        'Furniture.tsv': 792113,
        'Gift_Card.tsv': 149086, 
        'Grocery.tsv': 2402458, 
        'Health_Personal_Care.tsv': 5331449, 
        'Home.tsv': 6221559,
        'Home_Entertainment.tsv': 705889,
        'Home_Improvement.tsv': 2634781,
        'Jewelry.tsv': 1767753,
        'Kitchen.tsv': 4880466,
        'Lawn_and_Garden.tsv': 2557288,
        'Luggage.tsv': 348657,
        'Major_Appliances.tsv': 96901, 
        'Mobile_Apps.tsv': 5033376, 
        'Mobile_Electronics.tsv': 104975, 
        'Music.tsv': 4751577, 
        'Musical_Instruments.tsv': 904765,
        'Office_Products.tsv': 2642434,
        'Outdoors.tsv': 2302401,
        'PC.tsv': 6908554,
        'Personal_Care_Appliances.tsv': 85981,
        'Pet_Products.tsv': 2643619, 
        'Shoes.tsv': 4366916,
        'Software.tsv': 341931,
        'Sports.tsv': 4850360,
        'Tools.tsv': 1741100,
        'Toys.tsv': 4864249,
        'Video.tsv': 380604,
        'Video_DVD.tsv': 5069140,
        'Video_Games.tsv': 1785997, 
        'Watches.tsv': 960872, 
        'Wireless.tsv': 9002021
      }

if __name__ == "__main__":
    pass