__author__ = 'Yitao Sun'

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from Model import my_model, serving_fn


def load_data_set(file):
	df = pd.read_csv(file)
	msk = np.random.rand(len(df)) < 0.8
	return df[msk], df[~msk]

def one_hot(dataset, match):
	dataset['y'] = 0
	dataset.loc[dataset[match] == 'FAKE', 'y'] = 1
	return dataset

if __name__ == "__main__":

	tf.logging.set_verbosity(tf.logging.INFO)

	############################################ DATA LOADING ###############################
	train, test = load_data_set("./data/fake_or_real_news.csv")

	BASE_EXPORT_DIR = "./model_cache/"

	train = one_hot(train, 'label')
	test = one_hot(test, 'label')

	print(train.shape)
	print(test.shape)
	print(train.head())

	train_input_fn = tf.estimator.inputs.pandas_input_fn(
		train, train['y'], num_epochs=None, shuffle=True)

	predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
		test, test['y'], shuffle=False)

	predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
		train, train['y'], shuffle=False)

	############################################# MODELING ###################################
	embedded_text_feature_column = hub.text_embedding_column(
		key="text", 
		module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
		trainable=True)

	classifier = tf.estimator.Estimator(
		model_fn=my_model,
		params={
			'feature_columns': embedded_text_feature_column,
			# Two hidden layers of 10 nodes each.
			'hidden_units': [500, 100],
			# The model must choose between 2 classes.
			'n_classes': 2,
			'learning_rate': 0.03,
			'l2_regularization_strength':0.01
		})

	classifier.train(
		input_fn= train_input_fn,
		steps=1000)

	train_eval_result = classifier.evaluate(input_fn=predict_train_input_fn)
	test_eval_result = classifier.evaluate(input_fn=predict_test_input_fn)

	print("Training set accuracy: {accuracy}".format(**train_eval_result))
	print("Test set accuracy: {accuracy}".format(**test_eval_result))

	classifier.export_savedmodel(export_dir_base=BASE_EXPORT_DIR, serving_input_receiver_fn=serving_fn)
