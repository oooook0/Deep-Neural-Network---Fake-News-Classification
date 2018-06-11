__author__ = 'Yitao Sun'

import tensorflow as tf 


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

    export_outputs = {'predict_output': tf.estimator.export.PredictOutput({"pred_output_classes": predicted_classes[:, tf.newaxis], 'probabilities':tf.nn.softmax(logits)})}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar('loss', loss)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        eval_metrics_ops = {
            'accuracy': accuracy,
            'precision': tf.metrics.precision(labels=labels, predictions=predicted_classes),
            'recall': tf.metrics.recall(labels=labels, predictions=predicted_classes)
        }
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metrics_ops)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=params['learning_rate'], l2_regularization_strength=params['l2_regularization_strength'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, export_outputs=export_outputs)



def serving_fn():
    receiver_tensor = {
        "text": tf.placeholder(dtype=tf.string, shape=None)
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)