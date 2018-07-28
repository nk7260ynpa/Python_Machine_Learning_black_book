import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

df = pd.read_csv('movie_data.csv')
count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)

def preprocessor(text):
    text = re.sub('<[^>]*>', '',text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ''.join(emoticons).replace('-', '')
    return text

#print(df.loc[0, 'review'])
#print(preprocessor(df.loc[0, 'review']))
#print(preprocessor("</a>This :) is :(a test :-)!"))
#df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

#print('runners like running and thus they run')
porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

#print(tokenizer_porter('runners like running and thus they run'))

stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])

