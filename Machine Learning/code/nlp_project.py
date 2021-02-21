# importing necessary libraries
from fastai.text import *
from pandas_ods_reader import read_ods

# reading data
data = read_ods("hi_3500.ods", 1, headers=False)
#print(data.head(),data.shape[0])

# tabulate data
table = data.copy().sample(9077, random_state=0,replace=False)
table = table[['column_0', 'column_1']]
table.dropna(inplace=True)
#print(table[1:10])

# create a databunch for training the language model
data_lm = (TextList.from_df(table, cols='column_0').split_by_rand_pct(0.1).label_for_lm().databunch())

# create a dataclass for classification
data_clas = (TextList.from_df(table, cols='column_0', vocab=data_lm.vocab).split_by_rand_pct(0.1).label_from_df('column_1').databunch())

# display data
print(data_lm.show_batch(2))
print(data_clas.show_batch(1))
#data_lm.save('data_lm.pkl')
#data_lm.vocab.itos[:3]

# initialise learner from pretrained language model 
learn = language_model_learner(data_lm, AWD_LSTM, pretrained="pretrained_lm_hindi.pth", drop_mult=0.5)

# find the optimum learning rate
print(learn.lr_find())
print(learn.recorder.plot())

# fine tune the learners last layers
learn.fit_one_cycle(5, 2e-02, moms=(0.8, 0.7)) 

# unfreeze all the layers and train once more  
learn.unfreeze()
print(learn.lr_find())
print(learn.recorder.plot(skip_end=15))

# fine tuning the whole model
learn.fit_one_cycle(4, 2e-03, moms=(0.8, 0.7)) 

# test the language model
TEXT = " में "
N_WORDS = 30
N_SENTENCES = 1

print("\n".join(learn.predict(TEXT, N_WORDS, temperature=0.75) for _ in range(N_SENTENCES)))

# save the model and its encoder
# learn.export("models/hi_sentence.pkl")
# learn.save_encoder('fine_tuned_enc_9077')

# initialise the text classifier with the saved encoder
learn_clas = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
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
#learn_clas.export("classifier_sentiment_hi_v10.pkl")

# predict the sentiment 
pred=(learn_clas.predict("आज़तक पर देखिये कितनी स्मार्ट हुई स्मार्ट फेंसिंग Exclusive आज रात 9.30 pm पर @BSF_India"))
if int(pred[0])==2:
	print('positive')
elif int(pred[0])==0:
	print('negative')
elif int(pred[0])==1:
	print('neutral')
# validation accuracy: 0.821389

# ENSEMBLE METHOD

data_lm_bwd = (TextList.from_df(table, cols='column_0').split_by_rand_pct(0.1).label_for_lm().databunch(backwards=True))
data_clas_bwd = (TextList.from_df(table, cols='column_0', vocab=data_lm_bwd.vocab).split_by_rand_pct(0.1).label_from_df('column_1').databunch(backwards=True))

# show data
print(data_lm_bwd.show_batch(2))
print(data_clas_bwd.show_batch(2))

#save databunch
data_lm_bwd.save('data_lm_bwd.pkl')
#print(data_lm_bwd.vocab.itos[:3])

#initialise backwards language model
learn_bwd = language_model_learner(data_lm_bwd, AWD_LSTM, pretrained="pretrained_lm_hindi.pth", drop_mult=0.5)

#finding theoptimal parameters
learn_bwd.lr_find()
learn_bwd.recorder.plot()

learn_bwd.fit_one_cycle(3, 2e-02, moms=(0.8, 0.7)) 

learn_bwd.unfreeze()
learn_bwd.lr_find()
learn_bwd.recorder.plot(skip_end=15)

learn_bwd.fit_one_cycle(5, 2e-03, moms=(0.8, 0.7)) 

learn_bwd.export("models/hi_sentence_bwd.pkl")
learn_bwd.save_encoder('fine_tuned_enc_9077_bwd')

learn_clas_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, drop_mult=0.5)
learn_clas_bwd.load_encoder('fine_tuned_enc_9077_bwd')

learn_clas_bwd.lr_find()
learn_clas_bwd.recorder.plot()

learn_clas_bwd.fit_one_cycle(5, 2e-02, moms=(0.8,0.7)) 

learn_clas_bwd.freeze_to(-2)
learn_clas_bwd.fit_one_cycle(1, slice(1e-03/(2.6**4),1e-03), moms=(0.8,0.7))

learn_clas_bwd.freeze_to(-6)
learn_clas_bwd.fit_one_cycle(3, slice(5e-03/(2.6**4),5e-03), moms=(0.8,0.7))

preds,targs = learn_clas.get_preds(ordered=True)
print(accuracy(preds,targs)) # validation accuracy: 0.78

preds_b,targs_b = learn_clas_bwd.get_preds(ordered=True)
print(accuracy(preds_b,targs_b))

preds_avg = (preds+preds_b)/2

print(accuracy(preds_avg,targs_b)) # total validation accuracy: 0.78
