import multiprocessing
import os
import re
import sys
import time
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd

import librosa
import requests
from bs4 import BeautifulSoup
from keras import utils
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


########This part of the code contains functions for accuracy predictions########
def predict_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0],MFCCs.shape[1],MFCCs.shape[2],1)
    y_predicted = model.predict_classes(MFCCs,verbose=0)
    return(Counter(list(y_predicted)).most_common(1)[0][0])

def predict_class_all(X_train, model):
    '''
    :param X_train: List of segmented mfccs
    :param model: trained model
    :return: list of predictions
    '''
    predictions = []
    for mfcc in X_train:
        predictions.append(predict_class_audio(mfcc, model))
    return predictions

def confusion_matrix(y_predicted,y_test):
    '''
    Create confusion matrix
    :param y_predicted: list of predictions
    :param y_test: numpy array of shape (len(y_test), number of classes). 1.'s at index of actual, otherwise 0.
    :return: numpy array. confusion matrix
    '''
    confusion_matrix = np.zeros((len(y_test[0]),len(y_test[0])),dtype=int )
    for index, predicted in enumerate(y_predicted):
        confusion_matrix[np.argmax(y_test[index])][predicted] += 1
    return(confusion_matrix)

def get_accuracy(y_predicted,y_test):
    '''
    Get accuracy
    :param y_predicted: numpy array of predictions
    :param y_test: numpy array of actual
    :return: accuracy
    '''
    c_matrix = confusion_matrix(y_predicted,y_test)
    return( np.sum(c_matrix.diagonal()) / float(np.sum(c_matrix)))

########This part of the code will get the language metadata########
ROOT_URL = 'http://accent.gmu.edu/'
BROWSE_LANGUAGE_URL = 'browse_language.php?function=find&language={}'
WAIT = 1.2
DEBUG = True

def get_htmls(urls):
    '''
    Retrieves html in text form from ROOT_URL
    :param urls (list): List of urls from which to retrieve html
    :return (list): list of HTML strings
    '''
    htmls = []
    for url in urls:
        if DEBUG:
            print('downloading from {}'.format(url))
        htmls.append(requests.get(url).text)
        time.sleep(WAIT)

    return(htmls)


def build_search_urls(languages):
    '''
    creates url from ROOT_URL and languages
    :param languages (list): List of languages
    :return (list): List of urls
    '''
    return([ROOT_URL+BROWSE_LANGUAGE_URL.format(language) for language in languages])

def parse_p(p_tag):
    '''
    Extracts href property from HTML <p> tag string
    :param p_tag (str): HTML string
    :return (str): string of link
    '''
    text = p_tag.text.replace(' ','').split(',')
    return([ROOT_URL+p_tag.a['href'], text[0], text[1]])

def get_bio(hrefs):
    '''
    Retrieves HTML from list of hrefs and returns bio information
    :param hrefs (list): list of hrefs
    :return (DataFrame): Pandas DataFrame with bio information
    '''

    htmls = get_htmls(hrefs)
    bss = [BeautifulSoup(html,'html.parser') for html in htmls]
    rows = []
    bio_row = []
    for bs in bss:
        rows.append([li.text for li in bs.find('ul','bio').find_all('li')])
    for row in rows:
        bio_row.append(parse_bio(row))

    return(pd.DataFrame(bio_row))

def parse_bio(row):
    '''
    Parse bio data from row string
    :param row (str): Unparsed bio string
    :return (list): Bio columns
    '''
    cols = []
    for col in row:
        try:
            tmp_col = re.search((r"\:(.+)",col.replace(' ','')).group(1))
        except:
            tmp_col = col
        cols.append(tmp_col)
    return(cols)


def create_dataframe(languages):
    '''

    :param languages (str): language from which you want to get html
    :return df (DataFrame): DataFrame that contains all audio metadata from searched language
    '''
    htmls = get_htmls(build_search_urls(languages))
    bss = [BeautifulSoup(html,'html.parser') for html in htmls]
    persons = []

    for bs in bss:
        for p in bs.find_all('p'):
            if p.a:
                persons.append(parse_p(p))

    df = pd.DataFrame(persons, columns=['href','language_num','sex'])

    bio_rows = get_bio(df['href'])

    if DEBUG:
        print('loading finished')

    df['birth_place'] = bio_rows.iloc[:,0]
    df['native_language'] = bio_rows.iloc[:,1]
    df['other_languages'] = bio_rows.iloc[:,2]
    df['age_sex'] = bio_rows.iloc[:,3]
    df['age_of_english_onset'] = bio_rows.iloc[:,4]
    df['english_learning_method'] = bio_rows.iloc[:,5]
    df['english_residence'] = bio_rows.iloc[:,6]
    df['length_of_english_residence'] = bio_rows.iloc[:,7]

    df['birth_place'] = df['birth_place'].apply(lambda x: x[:-6].split(' ')[-2:])
    # print(df['birth_place'])
    # df['birth_place'] = lambda x: x[:-6].split(' ')[2:], df['birth_place']
    df['native_language'] = df['native_language'].apply(lambda x: x.split(' ')[2])
    # print(df['native_language'])
    # df['native_language'] = lambda x: x.split(' ')[2], df['native_language']
    df['other_languages'] = df['other_languages'].apply(lambda x: x.split(' ')[2:])
    # print(df['other_languages'])
    # df['other_languages'] = lambda x: x.split(' ')[2:], df['other_languages']
    df['age_sex'], df['age'] = df['age_sex'].apply(lambda x: x.split(' ')[2:]), df['age_sex'].apply(lambda x: x.replace('sex:','').split(',')[1])
    # print(df['age'])
    # df['age_sex'] = lambda x: x.split(' ')[2], df['age_sex']
    # df['age_of_english_onset'] = lambda x: float(x.split(' ')[-1]), df['age_of_english_onset']
    df['age_of_english_onset'] = df['age_of_english_onset'].apply(lambda x: float(x.split(' ')[-1]))
    # print(df['age_of_english_onset'])
    # df['english_learning_method'] = lambda x: x.split(' ')[-1], df['english_learning_method']
    df['english_learning_method'] = df['english_learning_method'].apply(lambda x: x.split(' ')[-1])
    # print(df['english_learning_method'])
    # df['english_residence'] = lambda x: x.split(' ')[2:], df['english_residence']
    df['english_residence'] = df['english_residence'].apply(lambda x: x.split(' ')[2:])
    # print(df['english_residence'])
    # df['length_of_english_residence'] = lambda x: float(x.split(' ')[-2]), df['length_of_english_residence']
    df['length_of_english_residence'] = df['length_of_english_residence'].apply(lambda x: float(x.split(' ')[-2]))
    # print(df['length_of_english_residence'])

    # df['age'] = lambda x: x.replace(' ','').split(',')[0], df['age_sex']

    return(df)

########This part of the code will download the sound files########
class GetAudio:

    def __init__(self, csv_filepath, destination_folder= 'audio/', wait= 1.5, debug=False ):
        '''
        Initializes GetAudio class object
        :param destination_folder (str): Folder where audio files will be saved
        :param wait (float): Length (in seconds) between web requests
        :param debug (bool): Outputs status indicators to console when True
        '''
        self.csv_filepath = csv_filepath
        self.audio_df = pd.read_csv(csv_filepath)
        self.url = 'http://chnm.gmu.edu/accent/soundtracks/{}.mp3'
        self.destination_folder = destination_folder
        self.wait = wait
        self.debug = False

    def check_path(self):
        '''
        Checks if self.distination_folder exists. If not, a folder called self.destination_folder is created
        '''
        if not os.path.exists(self.destination_folder):
            if self.debug:
                print('{} does not exist, creating'.format(self.destination_folder))
            os.makedirs('./' + self.destination_folder)

    def get_audio(self):
        '''
        Retrieves all audio files from 'language_num' column of self.audio_df
        If audio file already exists, move on to the next
        :return (int): Number of audio files downloaded
        '''

        self.check_path()

        counter = 0

        for lang_num in self.audio_df['language_num']:
            if not os.path.exists('./' + self.destination_folder +'{}.wav'.format(lang_num)):
                if self.debug:
                    print('downloading {}'.format(lang_num))
                (filename, headers) = urllib.request.urlretrieve(self.url.format(lang_num))
                sound = AudioSegment.from_mp3(filename)
                sound.export('./' + self.destination_folder + "{}.wav".format(lang_num), format="wav")
                counter += 1

        return counter

########This part of the code filters and displays the filters of the metadata########
# def filter_df(df):
#     '''
#     Function to filter audio files based on df columns
#     df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
#     english_residence,length_of_english_residence,native_language,other_languages,sex]
#     :param df (DataFrame): Full unfiltered DataFrame
#     :return (DataFrame): Filtered DataFrame
#     '''
#
#     # Example to filter arabic, mandarin, and english and limit to 73 audio files
#     arabic = df[df['native_language'] == 'arabic']
#     mandarin = df[df['native_language'] == 'mandarin']
#     english = df[df.native_language == 'english'][:73]
#     mandarin = mandarin[mandarin.length_of_english_residence < 10][:73]
#     arabic = arabic[arabic.length_of_english_residence < 10][:73]
#
#     df = english.append(arabic)
#     df = df.append(mandarin)
#
#
#
#     return df

def filter_df(df):
    '''
    Function to filter audio files based on df columns
    df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
    english_residence,length_of_english_residence,native_language,other_languages,sex]
    :param df (DataFrame): Full unfiltered DataFrame
    :return (DataFrame): Filtered DataFrame
    '''

    arabic = df[df.native_language == 'arabic']
    mandarin = df[df.native_language == 'mandarin']
    english = df[df.native_language == 'english']

    mandarin = mandarin[mandarin.length_of_english_residence < 10]
    arabic = arabic[arabic.length_of_english_residence < 10]

    df = df.append(english)
    df = df.append(arabic)
    df = df.append(mandarin)

    return df

def split_people(df,test_size=0.2):
    '''
    Create train test split of DataFrame
    :param df (DataFrame): Pandas DataFrame of audio files to be split
    :param test_size (float): Percentage of total files to be split into test
    :return X_train, X_test, y_train, y_test (tuple): Xs are list of df['language_num'] and Ys are df['native_language']
    '''


    return train_test_split(df['language_num'],df['native_language'],test_size=test_size,random_state=1234)

########This part of the code contains the training model information########
DEBUG = True
SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 10 #35#250

def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x],y))
    return utils.to_categorical(y, len(lang_dict))

def get_wav(language_num):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''

    y, sr = librosa.load('./audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return(librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))

def remove_silence(wav, thresh=0.04, chunk=5000):
    '''
    Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
    :param wav (np array): Wav array to be filtered
    :return (np array): Wav array with silence removed
    '''

    tf_list = []
    for x in range(len(wav) / chunk):
        if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
            tf_list.extend([True] * chunk)
        else:
            tf_list.extend([False] * chunk)

    tf_list.extend((len(wav) - len(tf_list)) * [False])
    return(wav[tf_list])

def normalize_mfcc(mfcc):
    '''
    Normalize mfcc
    :param mfcc:
    :return:
    '''
    mms = MinMaxScaler()
    return(mms.fit_transform(np.abs(mfcc)))

def make_segments(mfccs,labels):
    '''
    Makes segments of mfccs and attaches them to the labels
    :param mfccs: list of mfccs
    :param labels: list of labels
    :return (tuple): Segments with labels
    '''
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    '''
    Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
    :param mfcc (numpy array): MFCC array
    :return (numpy array): Segmented MFCC array
    '''
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    '''
    Creates segmented MFCCs from X_train
    :param X_train: list of MFCCs
    :return: segmented mfccs
    '''
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)


def train_model(X_train,y_train,X_validation,y_validation, batch_size=128): #64
    '''
    Trains 2D convolutional neural network
    :param X_train: Numpy array of mfccs
    :param y_train: Binary matrix based on labels
    :return: Trained model
    '''

    # Get row, column, and class sizes
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols, 1)
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1 )
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'training samples')

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    # Stops training if accuracy does not change at least 0.005 over 10 epochs
    es = EarlyStopping(monitor='acc', min_delta=.005, patience=10, verbose=1, mode='auto')

    # Creates log file for graphical interpretation using TensorBoard
    tb = TensorBoard(log_dir='../logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)

    # Image shifting
    datagen = ImageDataGenerator(width_shift_range=0.05)

    # Fit model using ImageDataGenerator
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS,
                        callbacks=[es,tb], validation_data=(X_validation,y_validation))

    return (model)

def save_model(model, model_filename):
    '''
    Save model to file
    :param model: Trained model to be saved
    :param model_filename: Filename
    :return: None
    '''
    model.save('../models/{}.h5'.format(model_filename))  # creates a HDF5 file 'my_model.h5'
