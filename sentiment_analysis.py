# -*- coding: utf-8 -*-
"""
# read in data and NLTK
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')  # Download the necessary resource for named entity chunking
nltk.download('words')  # Download the necessary resource for named entity chunking
import nltk
nltk.download('vader_lexicon')
!pip install transformers

# read in data

df=pd.read_csv('Reviews.csv')
print(df.shape)
df=df.head(500)
print(df.shape)

df

"""# quick EDA
"""

ax=df['Score'].value_counts().sort_index().plot(kind='bar',title='Count of review bt Star',figsize=(10,5))
ax.set_xlabel('Review Star')
plt.show()

"""# basic NLTK"""

example=df['Text'][50]
example

tokens=nltk.word_tokenize(example)
tokens[0:10]

tagged=nltk.pos_tag(tokens)
tagged[0:10]

entities = nltk.ne_chunk(tagged)
entities.pprint()

"""# VADER Seniment scoring

### this used as a "bag of word" approach:

* Stop word are removed
* each word in scored and combined to a totatl score


"""

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

sia.polarity_scores('this is very not good')

sia.polarity_scores(example)

res={}
for i, row in tqdm(df.iterrows(),total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid]=sia.polarity_scores(text)

vaders=pd.DataFrame(res).T
vaders=vaders.reset_index().rename(columns={'index':'Id'})
vaders=vaders.merge(df,how='left')

vaders.head()

sns.barplot(data=vaders,x='Score',y='compound')
ax = plt.gca()
ax.set_title("compound score star review")
plt.show()

fig,axs=plt.subplots(1,3,figsize=(15,3))
sns.barplot(data=vaders,x='Score',y='pos',ax=axs[0])
sns.barplot(data=vaders,x='Score',y='neu',ax=axs[1])
sns.barplot(data=vaders,x='Score',y='neg',ax=axs[2])
axs[0].set_title('positive')
axs[1].set_title('neutral')
axs[2].set_title('negative')
plt.tight_layout()
plt.show()

"""# Robert pretrained model
* use as model trained of a large

* Transformer model accounts for the workds but also the context related to other words.

"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

print(example)
sia.polarity_scores(example)

def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict

res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

"""# New Section"""

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

"""# Compare Score between models"""

results_df.columns

"""# Combine and compare"""

sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()



"""# Review Examples:

*   Positive 1-Star and Negative 5-Star Reviews

   Lets look at some examples where the model scoring and review score differ the most.



"""

results_df.query('Score == 1') \
.sort_values('roberta_pos', ascending=False)['Text'].values[0]

results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]

#nevative sentiment 5-star view

results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]


"""# the transformers pipeline

#### quick and easy to run sentiment predictions
"""

from transformers import pipeline

sent_pipeline=pipeline("sentiment-analysis")

sent_pipeline('ya i am happy !')

