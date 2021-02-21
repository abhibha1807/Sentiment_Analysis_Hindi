from fastai.text import *
from pandas_ods_reader import read_ods
import pandas as pd

def train(filename):
	table = pd.read_csv(filename)
	table = table[['column_0', 'column_1']]
	table.dropna(inplace=True)


	data_lm = (TextList.from_df(table, cols='column_0').split_by_rand_pct(0.1).label_for_lm().databunch())

	# create a dataclass for classification
	data_clas = (TextList.from_df(table, cols='column_0', vocab=data_lm.vocab).split_by_rand_pct(0.1).label_from_df('column_1').databunch())

	# display data
	# (data_lm.show_batch(2))
	# (data_clas.show_batch(1))
	data_lm.save('data_lm.pkl')
	#data_lm.vocab.itos[:3]

	# initialise learner from pretrained language model 
	learn = language_model_learner(data_lm, AWD_LSTM, pretrained="pretrained_lm_hindi.pth", drop_mult=0.5,callback_fns=[CSVLogger])

	# fine tune the learners last layers
	learn.fit_one_cycle(5, 2e-02, moms=(0.8, 0.7)) 

	# unfreeze all the layers and train once more  
	learn.unfreeze()

	# fine tuning the whole model
	learn.fit_one_cycle(4, 2e-03, moms=(0.8, 0.7)) 


	# save the model and its encoder
	learn.export("models/hi_sentence.pkl")
	learn.save_encoder('finetuned_encoder')

	
	# initialise the text classifier with the saved encoder
	learn_clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5,callback_fns=[CSVLogger])
	learn_clas.load_encoder('finetuned_encoder')

	# find the optimum learning rate
	learn_clas.lr_find()
	learn_clas.recorder.plot()

	# train the last layer
	learn_clas.fit_one_cycle(3, 2e-02, moms=(0.8,0.7)) 

	# gradual unfreezing of layers (last 2)
	learn_clas.freeze_to(-2)
	learn_clas.fit_one_cycle(1, slice(1e-03/(2.6**4),1e-03), moms=(0.8,0.7))

	# (last 3 layers)
	learn_clas.freeze_to(-3)
	learn_clas.fit_one_cycle(3, slice(5e-03/(2.6**4),5e-03), moms=(0.8,0.7))

	# unfreeze the whole model and train the classifier more
	learn_clas.unfreeze()
	learn_clas.fit_one_cycle(5, slice(5e-04/(2.6**4),5e-04), moms=(0.8,0.7))

	#save the classifier 
	learn_clas.export("classifier_sentiment_hi_v10.pkl")

train('file.csv')