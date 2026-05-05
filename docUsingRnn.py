# ==========================================
# EXPERIMENT 09
# Text Summarization using RNN (Seq2Seq)
# ==========================================

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================================
# STEP 1: SAMPLE DATASET (Small for Demo)
# ==========================================

documents = [
    "Artificial intelligence is transforming the world by enabling machines to learn and make decisions.",
    "Machine learning is a subset of artificial intelligence that focuses on building models from data.",
    "Deep learning uses neural networks with many layers to solve complex problems.",
    "Natural language processing allows computers to understand and process human language."
]

summaries = [
    "AI transforms world",
    "ML builds models",
    "Deep learning solves problems",
    "NLP understands language"
]

# Add start and end tokens
summaries = ["startseq " + s + " endseq" for s in summaries]

# ==========================================
# STEP 2: TOKENIZATION
# ==========================================

tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents + summaries)

vocab_size = len(tokenizer.word_index) + 1

# Convert text to sequences
doc_sequences = tokenizer.texts_to_sequences(documents)
sum_sequences = tokenizer.texts_to_sequences(summaries)

# Padding
max_doc_len = max(len(seq) for seq in doc_sequences)
max_sum_len = max(len(seq) for seq in sum_sequences)

encoder_input = pad_sequences(doc_sequences, maxlen=max_doc_len, padding='post')
decoder_input = pad_sequences(sum_sequences, maxlen=max_sum_len, padding='post')

# Decoder output (shifted)
decoder_output = np.zeros_like(decoder_input)

for i in range(len(sum_sequences)):
    for t in range(1, len(sum_sequences[i])):
        decoder_output[i, t-1] = decoder_input[i, t]

decoder_output = np.expand_dims(decoder_output, -1)

# ==========================================
# STEP 3: BUILD SEQ2SEQ MODEL
# ==========================================

latent_dim = 64

# Encoder
encoder_inputs = Input(shape=(max_doc_len,))
enc_emb = Embedding(vocab_size, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(max_sum_len,))
dec_emb = Embedding(vocab_size, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# STEP 4: TRAIN MODEL
# ==========================================

print("\nTraining model...\n")

model.fit(
    [encoder_input, decoder_input],
    decoder_output,
    batch_size=2,
    epochs=200,
    verbose=1
)

# ==========================================
# STEP 5: INFERENCE MODELS
# ==========================================

# Encoder model
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder setup
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

dec_emb2 = Embedding(vocab_size, latent_dim)(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(
    dec_emb2, initial_state=decoder_states_inputs
)

decoder_states = [state_h2, state_c2]
decoder_outputs2 = decoder_dense(decoder_outputs2)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs2] + decoder_states
)

# ==========================================
# STEP 6: GENERATE SUMMARY
# ==========================================

reverse_word_index = {i: word for word, i in tokenizer.word_index.items()}

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index['startseq']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '')

        if sampled_word == 'endseq' or len(decoded_sentence.split()) > max_sum_len:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        states_value = [h, c]

    return decoded_sentence.strip()

# ==========================================
# STEP 7: TEST SUMMARY
# ==========================================

print("\n--- SUMMARY RESULTS ---\n")

for i in range(len(documents)):
    input_seq = encoder_input[i].reshape(1, -1)
    summary = decode_sequence(input_seq)

    print("Original Text:")
    print(documents[i])
    print("Generated Summary:")
    print(summary)
    print("-" * 50)