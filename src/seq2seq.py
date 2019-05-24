from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, Activation, Lambda, Multiply, Add, Permute
from keras import backend as K
import keras
import numpy as np


from src.utils import one_hot_encode, add_one


class Seq2Seq:
    def __init__(self, num_encoder_tokens, num_decoder_tokens, start_token, end_token,
                 latent_dim=256, projection='one_hot', emb_dim=64, attention=None,
                 restore_path=None):
        self.num_encoder_tokens = num_encoder_tokens
        self.num_decoder_tokens = num_decoder_tokens

        self.start_token = start_token
        self.end_token = end_token

        if restore_path:
            self.train_model = load_model(restore_path + '/train_model.s2s.h5')
            self.encoder_model = load_model(restore_path + '/encoder_model.s2s.h5')
            self.decoder_model = load_model(restore_path + '/decoder_model.s2s.h5')
            return

        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None,), name='encoder_input')
        if projection == 'one_hot':
            embedding = Embedding(
                num_encoder_tokens+1,
                num_encoder_tokens,
                weights=[np.concatenate((np.zeros(shape=(1, num_encoder_tokens)), np.identity(num_encoder_tokens)))],
                trainable=False,
                mask_zero=True,
                name='encoder_embedding'
            )
        elif projection == 'word2vec':
            embedding = Embedding(
                num_encoder_tokens + 1,
                emb_dim,
                mask_zero=True,
                name='encoder_embedding'
            )
        else:
             raise Exception("projection method not recognized")
        encoder_emb = embedding(encoder_inputs)
        if attention:
            encoder = LSTM(latent_dim, return_state=True, return_sequences=True, name='encoder_lstm')
        else:
            encoder = LSTM(latent_dim, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = encoder(encoder_emb)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name='decoder_input')
        if projection == 'one_hot':
            embedding = Embedding(
                num_decoder_tokens+1,
                num_decoder_tokens,
                weights=[np.concatenate((np.zeros(shape=(1, num_decoder_tokens)), np.identity(num_decoder_tokens)))],
                trainable=False,
                mask_zero=True,
                name='decoder_embedding'
            )
        elif projection == 'word2vec':
            embedding = Embedding(
                num_decoder_tokens+1,
                emb_dim,
                mask_zero=True,
                name='decoder_embedding'
            )
        else:
            raise Exception("projection method not recognized")
        decoder_emb = embedding(decoder_inputs)
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_emb,
                                             initial_state=encoder_states)

        if attention:
            if attention == 'bahdanau':
                w1 = Dense(latent_dim, name='w1')
                w2 = Dense(latent_dim, name='w2')
                v = Dense(1, name='v')
                scores = v(
                    Activation('tanh', name='tanh')(
                        Add(name='add_encoder_decoder_projection')(
                            [w1(encoder_outputs),
                             Permute((2, 1))(w2(decoder_outputs))])))
                attention_weights = Activation('softmax', name='softmax_scores')(scores)
                context_vector = Multiply(name='multiply_weights_encoder_outputs')([attention_weights, encoder_outputs])
                context_vector = Lambda(lambda x: K.sum(x, axis=1), name='reduce_sum_context_vector')(context_vector)
                decoder_outputs = Add(name='add_decoder_outputs_context_vector')([decoder_outputs, context_vector])
            else:
                raise Exception('unsupported attention: ' + attention)

        # decoder_dense = Dense(self.num_decoder_tokens, activation=keras.activations.softmax, name='decoder_softmax')
        decoder_dense = Dense(self.num_decoder_tokens, activation=keras.activations.softmax, name='decoder_softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Define sampling models
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(latent_dim,), name='decoder_state_input_h')
        decoder_state_input_c = Input(shape=(latent_dim,), name='decoder_state_input_c')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_emb, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)

    def pre_process_encoder_input(self, input):
        # return one_hot_encode(input, self.max_encoder_seq_length, self.num_encoder_tokens)
        return add_one(input, self.max_encoder_seq_length)

    def pre_process_decoder_input(self, input):
        # return one_hot_encode(input, self.max_decoder_seq_length, self.num_decoder_tokens)
        return add_one(input, self.max_decoder_seq_length)

    def pre_process_decoder_target(self, input):
        return one_hot_encode(input, self.max_decoder_seq_length, self.num_decoder_tokens)
        # return add_one(input, self.max_decoder_seq_length)

    def train(self, encoder_input_seqs, decoder_input_seqs, decoder_target_seqs, batch_size=64, epochs=100):
        # Run training
        self.max_encoder_seq_length = max([len(seq) for seq in encoder_input_seqs])
        self.max_decoder_seq_length = max([len(seq) for seq in decoder_input_seqs])
        encoder_input_data = self.pre_process_encoder_input(encoder_input_seqs)
        print('encoder_input_data.shape: ', encoder_input_data.shape)
        decoder_input_data = self.pre_process_decoder_input(decoder_input_seqs)
        print('decoder_input_data.shape: ', decoder_input_data.shape)
        decoder_target_data = self.pre_process_decoder_target(decoder_target_seqs)
        print('decoder_target_data.shape: ', decoder_target_data.shape)
        self.train_model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        self.train_model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_split=0.2)
        # Save model
        self.train_model.save('train_model.s2s.h5')
        self.encoder_model.save('encoder_model.s2s.h5')
        self.decoder_model.save('decoder_model.s2s.h5')

    def encode(self, input_seq):
        # encoder_input_data = one_hot_encode(input_seq, self.max_encoder_seq_length, self.num_encoder_tokens)
        encoder_input_data = self.pre_process_encoder_input(input_seq)
        return self.encoder_model.predict(encoder_input_data)

    def decode(self, states_value):
        # Generate empty target sequence of length 1.
        # target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        # Populate the first token of target sequence with the start token.
        # target_seq[0, 0, self.start_token] = 1.
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.start_token + 1
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_seq = []
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            decoded_seq += [sampled_token_index]

            # Exit condition: either hit max length
            # or find stop token.
            if (sampled_token_index == self.end_token or
                    len(decoded_seq) > self.max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            # target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            # target_seq[0, 0, sampled_token_index] = 1.
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index + 1

            # Update states
            states_value = [h, c]

        return decoded_seq

    def predict(self, input_seq):
        # Encode the input as state vectors.
        states_value = self.encode(input_seq)
        prediction = self.decode(states_value)

        return prediction


