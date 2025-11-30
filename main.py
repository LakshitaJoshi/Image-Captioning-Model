import string # captions are string
import numpy as np
import os # dataset from there
from PIL import Image
from pickle import dump, load
import tensorflow as tf
import matplotlib.pyplot as plt 
from keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
# img captions will have different lengths, need to convert them in single length -> pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from keras.utils import to_categorical, get_file # for 1-hot encoding
from keras.layers import add
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout

# small library for seeing the progress of loops.
# tell the progress if nb is stuck or crash etc.
from tqdm import tqdm
tqdm().pandas()
import time

# # load text file into memory and read file
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# getting all images with their captions
def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        img, caption = caption.split('\t')
        if img[:-2] not in descriptions: # -2 we want name of the image
            descriptions[img[:-2]] = [ caption]
        else:
            descriptions[img[:-2]].append(caption)
    return descriptions

# Data Cleaning - lower casing, removing puntuations and words containing numbers
def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation) # remove punctuations
    for img, caps in captions.items(): # img as key, caps as items
        for i, img_caption in enumerate(caps):
            img_caption.replace("-", " ")
            desc = img_caption.split() # splitting caption into different words (desc is description)

            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc] # remove punctuations from each token
            desc = [word for word in desc if (len(word) > 1)] #removing hanging words 's and a
            desc = [word for word in desc if(word.isalpha())] # removing tokens with numeric values

            # convert back to string
            img_caption = ' '.join(desc)
            captions[img][i] = img_caption

    return captions
    

# Updating Vocabulary  (build vocabulary of all unique words)
def text_vocabulary(descriptions):
    vocab = set() 
    for key in descriptions.keys():
        [vocab.update(d.split()) for d in descriptions[key]]
    return vocab

# all descriptions in a file
def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + '\t' + desc)
    data= "\n".join(lines)
    file = open(filename, "w")
    file.write(data)
    file.close()

dataset_text = "Flickr8k_text"
dataset_images = "Flickr8k_Dataset"
# ===================Start==================
# we prepare our text data
 filename = dataset_text + "/Flickr8k.token.txt"
#loading the file that contains all data
#mapping them into descriptions dictionary img to 5 captions
 descriptions = all_img_captions(filename)
 print("Length of descriptions = ", len(descriptions))

# cleaning descriptions
clean_descriptions = cleaning_text(descriptions)
# building vocabulary
vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary = ", len(vocabulary))
#saving each description to file
 save_descriptions(clean_descriptions, "descriptions.txt")

def download_with_retry(url, filename, max_retries = 3):
    for attempt in range(max_retries):
        try:
            return get_file(filename, url)
        except Exception as e:
            if attempt == max_retries -1:
                raise e
            print(f"Download attempt failed")
            time.sleep(3)

weights_url = "https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5"

#include_top=False do not include the last layer
# pooling='avg' adds a layer to have a fixed size final layer
weights_path = download_with_retry(weights_url, 'xception_weights.h5')
model = Xception(include_top=False, pooling='avg', weights=weights_path) 

def extract_features(directory):
    features = {}
    valid_images = ['.jpg', '.jpeg', '.png']
    for img in tqdm(os.listdir(directory)):
        # Skip files that don't end with valid image extensions
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = directory +"/" + img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis=0) # adds 1 more direction -> ht, wd, channel (accepts images as a batch (1 img : 1 batch))
        image = image/127.5 # img in 0-2
        image = image - 1.0 # img in 0-1

        feature = model.predict(image)
        features[img] = feature
    return features

# 2084 feature cectors
features = extract_features(dataset_images)
dump(features, open("features.p", 'wb'))
# ============================================================================================================
features = load(open("features.p", 'rb'))

# loading the data -> img & captions
def load_photos(filename):
    file = load_doc(filename)
    photos = file.split("\n")[:-1]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(dataset_images, photo))]
    return photos_present

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words) < 1:
            continue

        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            # know where token start and end for making the predictions perfectly
            desc = '<start>' + " ".join(image_caption) + '<end>'
            descriptions[image].append(desc)
    return descriptions

def load_features(photos):
    all_features = load(open("features.p", "rb"))
    features = {k:all_features[k] for k in photos}
    print(features)
    return features

filename = dataset_text + "/" + "Flickr_8k.trainImages.txt"

train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

tokenizer = create_tokenizer(train_descriptions)

dump(tokenizer, open("tokenizer.p", "wb"))
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

# making desc so that they converted into a rec batch (fixed ht and wd). 
# padd the tokens (each desc) max len of word not character
def max_len(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)

max_len = max_len(train_descriptions)
print(max_len)

#create input-output sequence pairs from the image description.

#data generator, used by model.fit()
def data_generator(descriptions, features, tokenizer, max_len):
    def generator():
        while True:
            for key, description_list in descriptions.items():
                feature = features[key][0]
                input_image, input_sequence, output_word = create_sequences(tokenizer, max_len, description_list, feature)
                for i in range(len(input_image)):
                    yield {'input_1': input_image[i], 'input_2': input_sequence[i]}, output_word[i]
    
    # Define the output signature for the generator
    output_signature = (
        {
            'input_1': tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            'input_2': tf.TensorSpec(shape=(max_len,), dtype=tf.int32)
        },
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
    
    # Create the dataset
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )
    
    return dataset.batch(32)

def create_sequences(tokenizer, max_len, desc_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([desc])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            # encode output sequence
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

#You can check the shape of the input and output for your model
dataset = data_generator(train_descriptions, features, tokenizer, max_len)
for (a, b) in dataset.take(1):
    print(a['input_1'].shape, a['input_2'].shape, b.shape)
    break

from keras.utils import plot_model

# define the captioning model
def define_model(vocab_size, max_len):

    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = Input(shape=(max_len,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Merging both models
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)

    return model

# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_descriptions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_len)

model = define_model(vocab_size, max_len)
epochs = 10

def get_steps_per_epoch(train_descriptions):
    total_sequences = 0
    for img_captions in train_descriptions.values():
        for caption in img_captions:
            words = caption.split()
            total_sequences += len(words) - 1
    # Ensure at least 1 step, even if sequences < batch_size
    return max(1, total_sequences // 32)

# Update training loop
steps = get_steps_per_epoch(train_descriptions)

# making a directory models to save our models
os.mkdir("models3")


for i in range(last_epoch + 1, last_epoch +1 + epoch_to_train):
    print(f"training epoch {i}")
    dataset = data_generator(train_descriptions, train_features, tokenizer, max_len)
    model.fit(dataset, epochs=4, steps_per_epoch=steps, verbose=1)
    model.save("models3/model_" + str(i) + ".h5")
