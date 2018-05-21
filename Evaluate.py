import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd


def load_data_set(file):

	df = pd.read_csv("fake_or_real_news.csv")
	msk = np.random.rand(len(df)) < 0.8

	return df[msk], df[~msk]


def one_hot(dataset, match):

	dataset['y'] = 0
	dataset.loc[dataset[match] == 'FAKE', 'y'] = 1
	
	return dataset

def train_and_evaluate_with_module(hub_module, train_module=False):

  	embedded_text_feature_column = hub.text_embedding_column(
      key="text", module_spec=hub_module, trainable=train_module)

  	estimator = tf.estimator.DNNClassifier(
      hidden_units=[500, 100],
      feature_columns=[embedded_text_feature_column],
      n_classes=2,
      optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

 	estimator.train(input_fn=train_input_fn, steps=1000)

  	train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
  	test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

  	training_set_accuracy = train_eval_result["accuracy"]
  	test_set_accuracy = test_eval_result["accuracy"]

  	return {
      "Training accuracy": training_set_accuracy,
      "Test accuracy": test_set_accuracy
  	}

if __name__ == "__main__":

	tf.logging.set_verbosity(tf.logging.INFO)

	train, test = load_data_set("./data/fake_or_real_news.csv")

	train = one_hot(train, 'label')
	test = one_hot(test, 'label')

	print(train.shape)
	print(test.shape)
	print(train.head())

	print('Fake news percentage is {}.'.format( len(train[train['y'] ==1])/ len(train)) )


	train_input_fn = tf.estimator.inputs.pandas_input_fn(
		train, train['y'], num_epochs=None, shuffle=True)

	predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
	    train, train['y'], shuffle=False)

	predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
	    test, test['y'], shuffle=False)

	results = {}
	results["nnlm-en-dim128"] = train_and_evaluate_with_module(
	    "https://tfhub.dev/google/nnlm-en-dim128/1")
	results["nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
	    "https://tfhub.dev/google/nnlm-en-dim128/1", True)
	results["random-nnlm-en-dim128"] = train_and_evaluate_with_module(
	    "https://tfhub.dev/google/random-nnlm-en-dim128/1")
	results["random-nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
	    "https://tfhub.dev/google/random-nnlm-en-dim128/1", True)

	print(pd.DataFrame.from_dict(results, orient="index"))