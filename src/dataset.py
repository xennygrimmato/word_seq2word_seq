from __future__ import print_function


class Dataset:

    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.lines = f.read().split('\n')

    def generate_samples(self, num_samples=1000000):
        # Vectorize the data.
        self.input_texts = []
        self.target_texts = []
        input_characters = set()
        target_characters = set()
        for line in self.lines[: min(num_samples, len(self.lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)

        print('Number of samples:', len(self.input_texts))

        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

        self.encoder_input_seqs = [self.input_text_to_indices(text) for text in self.input_texts]
        self.decoder_input_seqs = [self.target_text_to_indices(text) for text in self.target_texts]
        self.decoder_target_seqs = [decoder_input_seq[1:] for decoder_input_seq in self.decoder_input_seqs]

    def input_text_to_indices(self, text):
        return [self.input_token_index[char] for char in text]

    def target_text_to_indices(self, text):
        return [self.target_token_index[char] for char in text]

    def input_indices_to_text(self, seq):
        return ''.join([self.reverse_input_char_index[idx] for idx in seq])

    def target_indices_to_text(self, seq):
        return ''.join([self.reverse_target_char_index[idx] for idx in seq])
