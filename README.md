# Deep Neural Network with Tensorflow Hub - Fake News Classification
This project classifies fake news using new Tensorflow word embeding from pre trained [Tensorflow Hub](https://www.tensorflow.org/hub/) models

About
=====
The fake news data of this project comes from [GeorgeMcIntire/fake_real_news_dataset](https://github.com/GeorgeMcIntire/fake_real_news_dataset)

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
    
    $ python Test.py

```
