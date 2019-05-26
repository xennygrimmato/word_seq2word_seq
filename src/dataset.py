from __future__ import print_function


class Dataset:

    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.lines = f.read().split('\n')

    def generate_samples(self, num_samples=1000000, character_level=False):
        # Vectorize the data.
        self.character_level = character_level
        self.input_texts = []
        self.target_texts = []
        input_tokens = set()
        target_tokens = set()
        for line in self.lines[: min(num_samples, len(self.lines) - 1)]:
            input_text, target_text = line.split('\t')
            # We use "tab" as the "start sequence" token
            # for the targets, and "\n" as "end sequence" token.
            target_text = self.join_tokens(['\t', target_text, '\n'])
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for token in self.split_tokens(input_text):
                if token not in input_tokens:
                    input_tokens.add(token)
            for token in self.split_tokens(target_text):
                if token not in target_tokens:
                    target_tokens.add(token)
        input_tokens = sorted(list(input_tokens))
        target_tokens = sorted(list(target_tokens))
        self.num_encoder_tokens = len(input_tokens)
        self.num_decoder_tokens = len(target_tokens)

        print('Number of samples:', len(self.input_texts))

        self.input_token_index = dict(
            [(token, i) for i, token in enumerate(input_tokens)])
        self.target_token_index = dict(
            [(token, i) for i, token in enumerate(target_tokens)])

        self.reverse_input_token_index = dict(
            (i, token) for token, i in self.input_token_index.items())
        self.reverse_target_token_index = dict(
            (i, token) for token, i in self.target_token_index.items())

        self.encoder_input_seqs = [self.input_text_to_indices(text) for text in self.input_texts]
        self.decoder_input_seqs = [self.target_text_to_indices(text) for text in self.target_texts]
        self.decoder_target_seqs = [decoder_input_seq[1:] for decoder_input_seq in self.decoder_input_seqs]

    def input_text_to_indices(self, text):
        return [self.input_token_index[token] for token in self.split_tokens(text)]

    def target_text_to_indices(self, text):
        return [self.target_token_index[token] for token in self.split_tokens(text)]

    def input_indices_to_text(self, seq):
        return self.join_tokens([self.reverse_input_token_index[idx] for idx in seq])

    def target_indices_to_text(self, seq):
        return self.join_tokens([self.reverse_target_token_index[idx] for idx in seq])

    def split_tokens(self, text):
        if self.character_level:
            return text
        else:
            return [t for t in text.split(' ') if t != '']

    def join_tokens(self, tokens):
        if self.character_level:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
