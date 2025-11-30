# Image-Captioning-Model Using CNN + LSTM

I built an image captioning model using the Flickr-8k dataset.
The model extracts visual features using a pre-trained <u> Xception network</u> and generates captions word-by-word using <u>LSTM decoder</u> trained on cleaned and <u>tokenized text data</u>.

## Features
- Text preprocessing, tokenization, and vocabulary building
- CNN-based feature extraction using Xception
- LST<-based caption generation
- Separate training and inference scripts (main.py + test.py)
- Ability to caption unseen images

## Tech Used
Python | TensorFlow | Xception | LSTM | Flickr-8k
