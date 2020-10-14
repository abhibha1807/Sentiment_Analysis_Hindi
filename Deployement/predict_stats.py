from flask import Flask, request, jsonify
from flask.logging import create_logger
from flask_cors import CORS, cross_origin
import json
import logging

from fastai.text import *
from pandas_ods_reader import read_ods

def predict_sentiment(test_str):

  l = load_learner('/media/gaurav/Study_Work/Engineering/Sem_6/NLP/project/content/', 'classifier_sentiment_hi_v10.pkl')

  pred=(l.predict(test_str))
  if int(pred[0])==2:
    return((float(max(list(pred[2])))),'positive')
    
  elif int(pred[0])==0:
    return((float(max(list(pred[0])))),'negative')
  elif int(pred[0])==1:
    return((float(max(list(pred[2])))),'neutral')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

LOG = create_logger(app)
LOG.setLevel(logging.INFO)


@app.route("/")
def home():
    html = f"<h3>Sklearn Prediction Home</h3>"
    return html.format(format)

@app.route("/predict", methods=['POST'])
def predict():
	LOG.info("Reuqest")
	LOG.info(request)
	json_payload = request.json
	LOG.info("JSON payload: \n%s",json_payload)
	LOG.info("Statement: %s",json_payload['statement'])
	prob,pred=predict_sentiment(json_payload['statement'])
	prediction = { "percent" : prob, "sentiment" : pred }
	return jsonify({'prediction': prediction})

@app.route("/contribute", methods=['POST'])
def contribute():
	LOG.info("Reuqest")
	LOG.info(request)
	json_payload = request.json
	LOG.info("JSON payload: \n%s",json_payload)
	with open('document.csv','a') as fd:
		fd.write(json.dumps(json_payload))
	return "1";

# test_str='आज़तक पर देखिये कितनी स्मार्ट हुई स्मार्ट फेंसिंग Exclusive आज रात 9.30 pm पर @BSF_India'
# prob=0.0
# pred=''
# prob,pred=predict_sentiment(test_str)
# print(prob)
# print(pred)

if __name__ == "__main__":
    # load pretrained model as clf
    app.run(host='0.0.0.0', port=8000, debug=True) # specify port=80
