# DNN with Tensorflow Hub - Fake News Classification

About
=====
This project classifies fake news using new Tensorflow word embeding from pre trained [Tensorflow Hub](https://www.tensorflow.org/hub/) models

## Data
The fake news data of this project comes from [GeorgeMcIntire/fake_real_news_dataset](https://github.com/GeorgeMcIntire/fake_real_news_dataset)

## Preprocessing
The preprocessing pipeline is entirely based on pretrained [google/nnlm-en-dim128/1](https://www.tensorflow.org/hub/modules/google/nnlm-en-dim128/1)

Token based text embedding trained on English Google News 200B corpus.

Text embedding based on feed-forward Neural-Net Language Models[1] with pre-built OOV. Maps from text to 128-dimensional embedding vectors.

## Neural Network Architecture
The model is built using [DNN Classifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) with 2 fully connected layers. 

## Results
The Fake News classifier was created to determine if a patient has retinopathy. The current model returns the following scores.


| Metric | Value |
| :-----: | :-----: |
| Accuracy (Train) | 100% |
| Accuracy (Test) | 95.16% |
| Precision | 100% |
| Recall | 100% |

## Dependencies

Python version: >= 3

Python packages needed inside `requirements.txt` 

## Instructions

1. Unzip the fake news dataset
```bash
    
    $ cd [path/to/dir]/data
    $ cd unzip fake_or_real_news.csv.zip

```
2. Install `requirements.txt` 
```bash
    
    $ pip install -r requirements.txt

```
3. Save `Tensorflow Hub` model to your current dir
```bash

    $ export TFHUB_CACHE_DIR=./my_module_cache

```
4. Run `Evaluate.py` to evaluate different word embeding preprocessor
```bash
    
    $ python Evaluate.py

```
5. Run `Train.py` to train the model the save it on `./model/` folder
```bash
    
    $ python Train.py --tensor_hub_model=[Tensorflow Hub Model to run] /
                      --max_steps=[Number of batches to run.] /
                      --base_export_dir=[Directory to save the model.]

```
6. Run `Test.py` to test your data on the newly trained model
```bash
    
    $ python Test.py --file_dir=[your json article input]

```

## Tensorlow Serving 

1. Note that tensorflow serving only works for model that is in the model version folder. Remember to place your model inside the folder that use model version as dir name
[Official documentation](https://www.tensorflow.org/serving/docker)

2. Download the docker image of `Tensorflow Serving` 
```bash
    
    $ docker pull tensorflow/serving

```
3. Run the docker `Tensorflow Serving` on port 8501(or other port you wanna open)
Note: /path/to/my_model/:the dir to the model dir(with the model version in it)
      /models/your_own_model_name: specify the model name you want
```bash
    
    $ docker run -p 8501:8501 \
        --mount type=bind,source=[/path/to/your_model/],target=[/models/your_own_model_name] \
        -e MODEL_NAME=[your_own_model_name] -t tensorflow/serving

```
4. Example using the Tensorflow serving API(RESTful Standard)
```python
import pprint
import requests as req 


data = '{"instances":["President Trump called the Iran nuclear deal a horrible agreement for the United States in response to Israeli Prime Minister Benjamin Netanyahus bombshell allegations about Tehrans covert activity – but stopped short of saying whether hed abandon the deal ahead of a looming deadline.  The president addressed the claims during a Rose Garden press conference Monday afternoon, moments after Netanyahu held a dramatic presentation revealing intelligence he says shows Iran is lying about its nuclear weapons program. That is just not an acceptable situation, Trump said. Trump said Netanyahu’s claims show Iran is not sitting back idly. Israeli prime minister claims Iran had been hiding all of the elements of a secret nuclear weapons program.Video Netanyahu on nuclear deal: Iran lied, big time Theyre setting off missiles which they say are for television purposes, Trump said. He added: I dont think so. Trump has repeatedly expressed a desire to exit the Iran deal, which was signed during the Obama administration. A crucial deadline for re-certifying the deal is on the horizon"]}'

response = req.post("http://localhost:8501/v1/models/my_model:predict", data = data.encode('utf-8'))

print(response.json())

```




