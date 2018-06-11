__author__ = 'Yitao Sun'

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('tensor_hub_model', 'https://tfhub.dev/google/nnlm-en-dim128/1',
					"""	define the word embeding model to use
					""")

tf.app.flags.DEFINE_integer('max_steps', 1000,
                            """Number of batches to run.""")

tf.app.flags.DEFINE_string('base_export_dir', os.getcwd() + '/model/',
                            """Directory to save the model.""")


def load_data_set(file):
	df = pd.read_csv(file)
	msk = np.random.rand(len(df)) < 0.8
	return df[msk], df[~msk]

def one_hot(dataset, match):
	dataset['y'] = 0
	dataset.loc[dataset[match] == 'FAKE', 'y'] = 1
	return dataset

def get_predictions(estimator, input_fn):

  	return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

def serving_fn():
    receiver_tensor = {
        "text": tf.placeholder(dtype=tf.string, shape=None)
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)

def main(argv=None):
	################################# Data Loading #######################################
	train, test = load_data_set("./data/fake_or_real_news.csv")

	train = one_hot(train, 'label')
	test = one_hot(test, 'label')

	print(train.shape)
	print(test.shape)
	print(train.head())

	print('Fake news percentage is {}.'.format(len(train[train['y'] ==1])/len(train)))
	
	################################## Modeling ##########################################

	embedded_text_feature_column = hub.text_embedding_column(
	    key="text", 
	    module_spec=FLAGS.tensor_hub_model,
	    trainable=True)

	# Transform Data to TF data type

	train_input_fn = tf.estimator.inputs.pandas_input_fn(
		train, train['y'], num_epochs=None, shuffle=True)

	predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
	    train, train['y'], shuffle=False)

	predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
	    test, test['y'], shuffle=False)


	estimator = tf.estimator.DNNClassifier(
	    hidden_units=[500, 100],
	    feature_columns=[embedded_text_feature_column],
	    n_classes=2,
	    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.003, l2_regularization_strength=0.01),
	    #model_dir=FLAGS.base_export_dir
	    )

	# Train model

	estimator.train(input_fn=train_input_fn, steps=FLAGS.max_steps)

	# Evalute model and get accuracy

	train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
	test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

	print("Training set accuracy: {accuracy}".format(**train_eval_result))
	print("Test set accuracy: {accuracy}".format(**test_eval_result))

	print(estimator.evaluate(input_fn=predict_test_input_fn)["accuracy_baseline"])

	estimator.export_savedmodel(export_dir_base=FLAGS.base_export_dir, serving_input_receiver_fn=serving_fn)
	
	################################confusion matrix######################################

	LABELS = [
	    "FAKE", "REAL"
	]

	# Create a confusion matrix on training data.

	with tf.Graph().as_default():
	  cm = tf.confusion_matrix(train["y"], 
	                           get_predictions(estimator, predict_train_input_fn))
	  with tf.Session() as session:
	    cm_out = session.run(cm)

	# Normalize the confusion matrix so that each row sums to 1.
	
	cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

	sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
	plt.xlabel("Predicted")
	plt.ylabel("True")
	plt.show()

if __name__ == "__main__":

	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run()
