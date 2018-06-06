__author__ = 'Yitao Sun'

import tensorflow as tf

from tensorflow.contrib import predictor

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', './model/1528292749',
                    """ define the model directory
                    """)

tf.app.flags.DEFINE_string('file_dir', None,
                    """ directory of the json file with "text" element to predict
                    """   )

prediction_fn = predictor.from_saved_model(export_dir=FLAGS.model_dir, signature_def_key='serving_default')


if __name__ == "__main__":

    if FLAGS.file_dir == None:
        predict_input_fn = {'inputs':["President Trump called the Iran nuclear deal a horrible agreement for the United States in response to Israeli Prime Minister Benjamin Netanyahus bombshell allegations about Tehrans covert activity – but stopped short of saying whether hed abandon the deal ahead of a looming deadline.  The president addressed the claims during a Rose Garden press conference Monday afternoon, moments after Netanyahu held a dramatic presentation revealing intelligence he says shows Iran is lying about its nuclear weapons program. That is just not an acceptable situation, Trump said. Trump said Netanyahu’s claims show Iran is not sitting back idly. Israeli prime minister claims Iran had been hiding all of the elements of a secret nuclear weapons program.Video Netanyahu on nuclear deal: Iran lied, big time Theyre setting off missiles which they say are for television purposes, Trump said. He added: I dont think so. Trump has repeatedly expressed a desire to exit the Iran deal, which was signed during the Obama administration. A crucial deadline for re-certifying the deal is on the horizon"]}
    else:
        predict_input_fn = json.loads(open(FLAGS.file_dir).read())

    output = prediction_fn(predict_input_fn)

    print(output['scores'])