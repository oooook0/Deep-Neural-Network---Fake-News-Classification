import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd 
import numpy as np 

tf.logging.set_verbosity(tf.logging.INFO)

def load_data_set(file):
	df = pd.read_csv(file)
	msk = np.random.rand(len(df)) < 0.8
	return df[msk], df[~msk]

def one_hot(dataset, match):
	dataset['True'] = 0
	dataset['False'] = 0
	dataset.loc[dataset[match] == 'FAKE', 'False'] = 1
	dataset.loc[dataset[match] != 'FAKE', 'True'] = 1
	return dataset

def fetch_batch(data_X, data_Y, batch_index, batch_size):

    train_X = data_X[batch_index*batch_size: (batch_index+1)*batch_size]
    train_Y = data_Y[batch_index*batch_size: (batch_index+1)*batch_size]

    return train_X, train_Y

if __name__ == "__main__":

	train, test = load_data_set("../data/fake_or_real_news.csv")

	train = one_hot(train, 'label')

	learning_rate = 0.03
	batch_size = 100
	BETA = 0.1

	n_batches = int(len(test)/batch_size)

	print(n_batches)

	with tf.Graph().as_default():

		embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim128/1")

		X = tf.placeholder(tf.float32, [batch_size, 128])
		Y = tf.placeholder(tf.float32, [batch_size, 2])
		w = tf.Variable(tf.random_normal(shape=[128, 2], stddev = 0.01), name = "weights")
		b = tf.Variable(tf.zeros([1,2]), name = "bias")

		logits = tf.matmul(X, w) + b

		entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits =logits, labels=Y)

		regularizer = tf.nn.l2_loss(w)

		loss = tf.reduce_mean(entropy + BETA*regularizer)

		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

		init = tf.global_variables_initializer()

		table = tf.tables_initializer()

		with tf.Session() as sess:
		    init.run()
		    table.run()

		    train_X = embed(list(train['text'])).eval()
		    train_Y = np.asarray(train[['False', 'True']].values)

		    total_correct_preds = 0

		    for i in range(n_batches):
		        X_batch, Y_batch = fetch_batch(train_X, train_Y, i, batch_size)
		        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],feed_dict={X: X_batch, Y:Y_batch})

		        preds = tf.nn.softmax(logits_batch)
		        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))

		        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
		        total_correct_preds += sess.run(accuracy)
		        print(total_correct_preds)
