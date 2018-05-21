import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


def load_data_set(file):
	df = pd.read_csv(file)
	msk = np.random.rand(len(df)) < 0.8
	return df[msk], df[~msk]

def one_hot(dataset, match):
	dataset['y'] = 0
	dataset.loc[dataset[match] == 'FAKE', 'y'] = 1
	return dataset

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.03 probability."""
    # Create three fully connected layers each layer having a dropout
    # probability of 0.1.

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

if __name__ == "__main__":

	tf.logging.set_verbosity(tf.logging.INFO)

	############################################ DATA LOADING ###############################
	train, test = load_data_set("./data/fake_or_real_news.csv")

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
	    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

	classifier = tf.estimator.Estimator(
	    model_fn=my_model,
	    params={
	        'feature_columns': [embedded_text_feature_column],
	        # Two hidden layers of 10 nodes each.
	        'hidden_units': [500, 100],
	        # The model must choose between 3 classes.
	        'n_classes': 2,
	        'learning_rate': 0.03
	    })

	classifier.train(
	    input_fn= train_input_fn,
	    steps=1000)

	train_eval_result = classifier.evaluate(input_fn=predict_train_input_fn)
	test_eval_result = classifier.evaluate(input_fn=predict_test_input_fn)

	print("Training set accuracy: {accuracy}".format(**train_eval_result))
	print("Test set accuracy: {accuracy}".format(**test_eval_result))

