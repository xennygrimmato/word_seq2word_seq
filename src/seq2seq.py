from __future__ import print_function

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, LSTM, Dense
import numpy as np


class Seq2SeqModel:
    def __init__(self, num_encoder_tokens, num_decoder_tokens, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length, latent_dim=256):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.max_encoder_seq_length = max_encoder_seq_length
        self.max_decoder_seq_length = max_decoder_seq_length



        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, num_encoder_tokens))
        encoder = LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

        # Reverse-lookup token index to decode sequences back to
        # something readable.
        self.reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())

    def train(self, encoder_input_data, decoder_input_data, decoder_target_data, batch_size=64, epochs=100):
        # Run training
        self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)
        # Save model
        self.train_model.save('train_model.s2s.h5')
        self.encoder_model.save('encoder_model.s2s.h5')
        self.decoder_model.save('decoder_model.s2s.h5')

    def encode_sequence(self, input_seq):
        return self.encoder_model.predict(input_seq)

    def decode_sequence(self, states_value):
        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
                    len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence

    def predict(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encode_sequence(input_seq)
        prediction = self.decode_sequence(states_value)

        return prediction


