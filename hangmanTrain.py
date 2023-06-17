import json
import requests
import random
import string
import secrets
import time
import re
import collections
import numpy as np
import os, pathlib, shutil

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import TextVectorization
from datasets import Dataset
from sklearn.ensemble import RandomForestClassifier


try:
    from urllib.parse import parse_qs, urlencode, urlparse
except ImportError:
    from urlparse import parse_qs, urlparse
    from urllib import urlencode

from requests.packages.urllib3.exceptions import InsecureRequestWarning

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config
    
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config

class HangmanAPI(object):
    def __init__(self, access_token=None, session=None, timeout=None):
        #self.hangman_url = self.determine_hangman_url()
        self.hangman_url = 0
        self.access_token = access_token
        self.session = session or requests.Session()
        self.timeout = timeout
        self.guessed_letters = []
        
        full_dictionary_location = "words_250000_train.txt"
        self.full_dictionary = self.build_dictionary(full_dictionary_location)        
        self.full_dictionary_common_letter_sorted = collections.Counter("".join(self.full_dictionary)).most_common()
        
        self.current_dictionary = []

        self.clf = []

        self.max_data_length = 10e6     # max samples of generated data
        self.num_passes = 3            # iterations over training dictionary
        self.max_length = 60            # max length for vectorized input
        self.batch_size = 32
        self.error_rate = 0.35          # guess an incorrect letter with probability=error_rate
        self.vocab_size = 29            # Transformer and Positional Embedding params
        self.embed_dim = 128
        self.num_heads = 2
        self.dense_dim = 16
        
    @staticmethod
    def determine_hangman_url():
        links = ['https://trexsim.com', 'https://sg.trexsim.com']

        data = {link: 0 for link in links}

        for link in links:

            requests.get(link)

            for i in range(10):
                s = time.time()
                requests.get(link)
                data[link] = time.time() - s

        link = sorted(data.items(), key=lambda x: x[1])[0][0]
        link += '/trexsim/hangman'
        return link

    def guess(self, word): # word input example: "_ p p _ e "
        ###############################################
        # Replace with your own "guess" function here #
        ###############################################

        # clean the word so that we strip away the space characters
        # replace "_" with "." as "." indicates any character in regular expressions

        clean_word = word.replace(" ", "")
        
        # find length of passed word
        len_word = len(clean_word)
        
        # grab current dictionary of possible words from self object, initialize new possible words dictionary to empty
        current_dictionary = self.current_dictionary
        new_dictionary = []

        model = keras.models.load_model("full_transformer_encoder.keras", custom_objects={"TransformerEncoder": TransformerEncoder, "PositionalEmbedding": PositionalEmbedding})


        clean_word += ' '
        for letter in self.guessed_letters:
            clean_word += letter

        # vectorize the input
        input = np.zeros((1, 60), dtype="int32")

        pos = 0
        for letter in clean_word:
            if letter == '_':
                input[0, pos] = 27
            elif letter == ' ':
                input[0, pos] = 28
            else:
                input[0, pos] = ord(letter) - 96
            pos += 1

        input = tf.convert_to_tensor(input, dtype=tf.int32)

        #output = model.predict(input)
        output = np.squeeze(np.array(self.clf.predict_proba(input)))[:, 1]

        

        guess_letter = '!'
        while True:
            ind = np.argmax(output)
            guess_letter = chr(ord('a') + ind)

            if guess_letter in self.guessed_letters:
                output[ind] = 0
            else:
                break


        
        '''
        # iterate through all of the words in the old plausible dictionary
        for dict_word in current_dictionary:
            # continue if the word is not of the appropriate length
            if len(dict_word) != len_word:
                continue
                
            # if dictionary word is a possible match then add it to the current dictionary
            if re.match(clean_word,dict_word):
                new_dictionary.append(dict_word)
        
        # overwrite old possible words dictionary with updated version
        self.current_dictionary = new_dictionary
        
        # count occurrence of all characters in possible word matches
        full_dict_string = "".join(new_dictionary)
        
        c = collections.Counter(full_dict_string)
        sorted_letter_count = c.most_common()                   
        
        guess_letter = '!'
        
        # return most frequently occurring letter in all possible words that hasn't been guessed yet
        for letter,instance_count in sorted_letter_count:
            if letter not in self.guessed_letters:
                guess_letter = letter
                break
            
        # if no word matches in training dictionary, default back to ordering of full dictionary
        if guess_letter == '!':
            sorted_letter_count = self.full_dictionary_common_letter_sorted
            for letter,instance_count in sorted_letter_count:
                if letter not in self.guessed_letters:
                    guess_letter = letter
                    break     

        '''       
        
        return guess_letter

    ############################################example##############
    # You'll likely not need to modify any of the code below #
    ##########################################################
    
    def build_dictionary(self, dictionary_file_location):
        text_file = open(dictionary_file_location,"r")
        full_dictionary = text_file.read().splitlines()
        text_file.close()
        return full_dictionary
                
    def start_game_train(self, practice=True, verbose=True):
        # reset guessed letters to empty set and current plausible dictionary to the full dictionary
        self.guessed_letters = []
        self.current_dictionary = self.full_dictionary
                         
        game_id = 0
        solution = random.choice(self.current_dictionary)

        word = '_' * len(solution)

        tries_remains = 6
        if verbose:
            print("Successfully start a new game! Game ID: {0}. # of tries remaining: {1}. Word: {2}.".format(game_id, tries_remains, word))
        while tries_remains>0:
            # get guessed letter from user code
            guess_letter = self.guess(word)
                
            # append guessed letter to guessed letters field in hangman object
            self.guessed_letters.append(guess_letter)
            if verbose:
                print("Guessing letter: {0}".format(guess_letter))
                
            '''
            TODO:
            1. Check if letter is contained in solution
            2. If yes, add letter to word and guessed letters
            3. If no, just add to guessed letters and tries_remains -= 1
            '''

            if guess_letter in solution:
                res = [i.start() for i in re.finditer(guess_letter, solution)]
                for ind in res:
                    if ind == len(solution) - 1:
                        word = word[:ind] + guess_letter
                    else:
                        word = word[:ind] + guess_letter + word[ind + 1:]
                if '_' not in word:
                    return True
            else:
                tries_remains -= 1

        return
        
    def my_status(self):
        return self.request("/my_status", {})
    
    def request(
            self, path, args=None, post_args=None, method=None):
        if args is None:
            args = dict()
        if post_args is not None:
            method = "POST"

        # Add `access_token` to post_args or args if it has not already been
        # included.
        if self.access_token:
            # If post_args exists, we assume that args either does not exists
            # or it does not need `access_token`.
            if post_args and "access_token" not in post_args:
                post_args["access_token"] = self.access_token
            elif "access_token" not in args:
                args["access_token"] = self.access_token

        time.sleep(0.2)

        num_retry, time_sleep = 50, 2
        for it in range(num_retry):
            try:
                response = self.session.request(
                    method or "GET",
                    self.hangman_url + path,
                    timeout=self.timeout,
                    params=args,
                    data=post_args,
                    verify=False
                )
                break
            except requests.HTTPError as e:
                response = json.loads(e.read())
                raise HangmanAPIError(response)
            except requests.exceptions.SSLError as e:
                if it + 1 == num_retry:
                    raise
                time.sleep(time_sleep)

        headers = response.headers
        if 'json' in headers['content-type']:
            result = response.json()
        elif "access_token" in parse_qs(response.text):
            query_str = parse_qs(response.text)
            if "access_token" in query_str:
                result = {"access_token": query_str["access_token"][0]}
                if "expires" in query_str:
                    result["expires"] = query_str["expires"][0]
            else:
                raise HangmanAPIError(response.json())
        else:
            raise HangmanAPIError('Maintype was not text, or querystring')

        if result and isinstance(result, dict) and result.get("error"):
            raise HangmanAPIError(result)
        return result
    
    def generate_dataset(self, dict):
        inputs = np.zeros((int(self.max_data_length), self.max_length), dtype="int32")
        targets = np.zeros((int(self.max_data_length), 26), dtype="int32")

        counter = 0
        est = 0
        for i in range(self.num_passes):
            for solution in dict:
                est += 1
                solution_partial = solution
                word = '_' * len(solution)
                word_aug = word + ' '
                while '_' in word:
                    # Add input example to dataset
                    pos = 0
                    for letter in word_aug:
                        if letter == '_':
                            inputs[counter, pos] = 27
                        elif letter == ' ':
                            inputs[counter, pos] = 28
                        else:
                            inputs[counter, pos] = ord(letter) - 96 # let zero be mask value
                        pos += 1

                    # Add target example to dataset
                    for letter in solution_partial:
                        ind = ord(letter) - 97
                        targets[counter, ind] = 1
                    counter += 1
                    if counter % 100000 == 0:
                        print('{} / {} (estimated)'.format(counter, int(counter / est * len(dict) * self.num_passes)))

                    # try to mess up a guess with probability = error_rate
                    if np.random.uniform(low=0, high=1) < self.error_rate:
                        r = random.randrange(26)
                        letter = chr(ord('a') + r)
                        if letter not in solution and letter not in word_aug:
                            word_aug += letter
                            continue
                        
                    # remove letter from solution set
                    r=random.randrange(len(solution_partial))
                    letter = solution_partial[r]
                    res = [i.start() for i in re.finditer(letter, solution_partial)]
                    for ind in np.flip(res):
                        if ind == len(solution_partial) - 1:
                            solution_partial = solution_partial[:ind]
                        else:
                            solution_partial = solution_partial[:ind] + solution_partial[ind + 1:]

                    # add to guess word
                    res = [i.start() for i in re.finditer(letter, solution)]
                    for ind in res:
                        if ind == len(solution) - 1:
                            word = word[:ind] + letter
                        else:
                            word = word[:ind] + letter + word[ind + 1:]
                        word_aug = word_aug[:ind] + letter + word_aug[ind + 1:]

                    word_aug += letter

        max_ind = np.floor(counter / self.batch_size)
        inputs = tf.convert_to_tensor(inputs[:int(max_ind * self.batch_size), :], dtype=tf.int32)
        targets = tf.convert_to_tensor(targets[:int(max_ind * self.batch_size), :], dtype=tf.int32)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        int_train_ds = dataset.batch(self.batch_size)

        return int_train_ds
    
class HangmanAPIError(Exception):
    def __init__(self, result):
        self.result = result
        self.code = None
        try:
            self.type = result["error_code"]
        except (KeyError, TypeError):
            self.type = ""

        try:
            self.message = result["error_description"]
        except (KeyError, TypeError):
            try:
                self.message = result["error"]["message"]
                self.code = result["error"].get("code")
                if not self.type:
                    self.type = result["error"].get("type", "")
            except (KeyError, TypeError):
                try:
                    self.message = result["error_msg"]
                except (KeyError, TypeError):
                    self.message = result

        Exception.__init__(self, self.message)


api = HangmanAPI(access_token="90bc065bfc8969d9beb0e11915a238", timeout=2000)

partition = np.random.choice(api.full_dictionary, int(len(api.full_dictionary) * 0.9), replace=False)
train = np.random.choice(partition, int(len(partition) * 0.8 / 0.9), replace=False)
val = np.setdiff1d(api.full_dictionary, partition)
test = np.setdiff1d(partition, train)

# Generate datasets
int_train_ds = api.generate_dataset(train)
int_val_ds = api.generate_dataset(val)
int_test_ds = api.generate_dataset(test)

inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(api.max_length, api.vocab_size, api.embed_dim)(inputs)
x = TransformerEncoder(api.embed_dim, api.dense_dim, api.num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(26, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

# train the model
callbacks = [keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras", save_best_only=True)]
#model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)
model = keras.models.load_model("full_transformer_encoder.keras", custom_objects={"TransformerEncoder": TransformerEncoder, "PositionalEmbedding": PositionalEmbedding})
#print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

count = 0
wins = 0
for i in range(100):
    if api.start_game_train(practice=1,verbose=True):
        wins +=1 
    count += 1
    print('{} / {}'.format(wins, count))

[total_practice_runs,total_recorded_runs,total_recorded_successes,total_practice_successes] = api.my_status() # Get my game stats: (# of tries, # of wins)
practice_success_rate = total_practice_successes / total_practice_runs
print('run %d practice games out of an allotted 100,000. practice success rate so far = %.3f' % (total_practice_runs, practice_success_rate))