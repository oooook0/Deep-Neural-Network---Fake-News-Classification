import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import os

def make_estimator(model_diry):

    embedded_text_feature_column = hub.text_embedding_column(key="text",module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    return tf.estimator.DNNClassifier(
        n_classes=2,
        feature_columns=[embedded_text_feature_column],
        hidden_units=[500, 100],
        model_dir=model_diry)

def data_loading(data):

    df = pd.DataFrame(data=predict_input_fn, index=[0])

    print(df.head())

    return tf.estimator.inputs.pandas_input_fn(x = df, shuffle=False)

def get_prediction(estimator, input_fn):

    predictions = estimator.predict(input_fn=input_fn)

    return list(predictions)[0]['probabilities']

if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.ERROR)

    MODEL_DIR = os.getcwd() + "/tmp/"

    predict_input_fn = {'text':"President Trump called the Iran nuclear deal a horrible agreement for the United States in response to Israeli Prime Minister Benjamin Netanyahus bombshell allegations about Tehrans covert activity – but stopped short of saying whether hed abandon the deal ahead of a looming deadline.  The president addressed the claims during a Rose Garden press conference Monday afternoon, moments after Netanyahu held a dramatic presentation revealing intelligence he says shows Iran is lying about its nuclear weapons program. That is just not an acceptable situation, Trump said. Trump said Netanyahu’s claims show Iran is not sitting back idly. Israeli prime minister claims Iran had been hiding all of the elements of a secret nuclear weapons program.Video Netanyahu on nuclear deal: Iran lied, big time Theyre setting off missiles which they say are for television purposes, Trump said. He added: I dont think so. Trump has repeatedly expressed a desire to exit the Iran deal, which was signed during the Obama administration. A crucial deadline for re-certifying the deal is on the horizon"}

    
    estimator_from_file = make_estimator(MODEL_DIR)

    train_input_fn = data_loading(predict_input_fn)

    predictions = get_prediction(estimator_from_file,train_input_fn)

    print("probability: {}".format(predictions))