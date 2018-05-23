# Deep Neural Network with Tensorflow Hub - Fake News Classification

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
The model is built using [DNN Classifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier). 

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
3. Run `Evaluate.py` to evaluate different word embeding preprocessor
```bash
    
    $ python Evaluate.py

```
4. Run `Train.py` to train the model the save it on `./tmp/` folder
```bash
    
    $ python Train.py

```
5. Run `Test.py` to test your data on the newly trained model
```bash
    
    $ python Test.py --file_dir=[your json article input]

```




