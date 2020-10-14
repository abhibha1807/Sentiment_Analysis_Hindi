from fastai.text import *
from pandas_ods_reader import read_ods

def predict_sentiment(test_str):

  l = load_learner('/content/', 'classifier_sentiment_hi_v10.pkl')

  pred=(l.predict(test_str))
  if int(pred[0])==2:
    return((float(max(list(pred[2])))),'positive')
    
  elif int(pred[0])==0:
    return((float(max(list(pred[0])))),'negative')
  elif int(pred[0])==1:
    return((float(max(list(pred[2])))),'neutral')

test_str='आज़तक पर देखिये कितनी स्मार्ट हुई स्मार्ट फेंसिंग Exclusive आज रात 9.30 pm पर @BSF_India'
prob=0.0
pred=''
prob,pred=predict_sentiment(test_str)