# Necessary imports
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import rnn
import re
import time


# -------------------------------------------------------MODEL-----------------------------------------------------------
# Creating placeholders for models inputs
def model_inputs():
    input_data = tf.placeholder(shape=[None, None], dtype=tf.int32, name='input')
    targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='targets')
    learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
    # For dropout node keeping probability
    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    return input_data, targets, learning_rate, keep_prob


# Creating the embeddings for our decoding layer
def process_encoding_input(target_data, vocab_to_int, batch_size):
    # Remove the last word from each sentence
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    # inserting a column of <GO> tag to the starting of the batch processes above
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)
    return dec_input


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.LSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(cell=lstm, input_keep_prob=keep_prob)
    enc_cell = tf.nn.rnn_cell.MultiRNNCell([drop] * num_layers)
    encoder_outputs, encoder_final_state = \
        tf.nn.bidirectional_dynamic_rnn(
            cell_fw=enc_cell,
            cell_bw=enc_cell,
            inputs=rnn_inputs,
            sequence_length=sequence_length,
            dtype=tf.float32
        )
    # encoder_outputs = tf.concat([encoder_fw_outputs,encoder_bw_outputs],2)
    # encoder_final_state_c = tf.concat([encoder_fw_final_state.c,encoder_bw_final_state.c],1)
    # encoder_final_state_h = tf.concat([encoder_fw_final_state.h,encoder_bw_final_state.h],1)
    # encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_final_state_c,h = encoder_final_state_h)
    return encoder_outputs, encoder_final_state


# Implementing our decoding layer with attention for training
def decoding_layer_train(encoder_outputs, encoder_state, dec_cell, rnn_size, dec_embed_input, sequence_length,
                         decoding_scope, vocab_size):
    attention_states = encoder_outputs
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, attention_states,
                                                               memory_sequence_length=sequence_length)
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size=rnn_size)
    helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input, sequence_length, rnn_size)
    projection_layer = tf.python.layers.core.Dense(vocab_size, use_bias=False, name="output_projection")
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell, helper=helper, initial_state=encoder_state,
                                              output_layer=projection_layer)
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, scope=decoding_scope)
    return outputs


# Implementing our decoding layer with attention for inference
def decoding_layer_infer(encoder_outputs, encoder_state, dec_cell, rnn_size, dec_embeddings, sequence_length,
                         decoding_scope, vocab_to_int, batch_size, vocab_size):
    attention_states = encoder_outputs
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(rnn_size, attention_states,
                                                               memory_sequence_length=sequence_length)
    dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell, attention_mechanism, attention_layer_size=rnn_size)
    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,
                                                      start_tokens=tf.tile(vocab_to_int['<GO>'], batch_size),
                                                      end_token=vocab_to_int['<EOS>'])
    projection_layer = tf.python.layers.core.Dense(vocab_size, use_bias=False, name="output_projection")
    decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell, helper=helper, initial_state=encoder_state,
                                              output_layer=projection_layer)
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, scope=decoding_scope)
    return outputs


# Final decoding layer
def decoding_layer(dec_embed_input, dec_embeddings, encoder_outputs, encoder_state, vocab_size, sequence_length,
                   rnn_size, num_layers, vocab_to_int, keep_prob, batch_size):
    with tf.variable_scope('decoding') as decoding_scope:
        lstm = tf.contrib.rnn.LSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        train_logits = decoding_layer_train(encoder_outputs, encoder_state, dec_cell, rnn_size, dec_embed_input,
                                            sequence_length, decoding_scope, vocab_size)

        decoding_scope.reuse_variables()

        infer_logits = decoding_layer_infer(encoder_outputs, encoder_state, dec_cell, rnn_size, dec_embeddings,
                                            sequence_length, decoding_scope, vocab_to_int, batch_size, vocab_size)

        return train_logits, infer_logits


# Final Seq2Seq Model
def seq2seq(
        input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size, questions_vocab_size,
        enc_embedding_size, dec_embedding_size, rnn_size, num_layers, questions_vocab_to_int):
    enc_embed_input = tf.contrib.layers.embed_sequence(input_data, answers_vocab_size + 1, enc_embedding_size,
                                                       initializer=tf.random_uniform_initializer(-1, 1))
    encoder_outputs, encoder_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)
    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size + 1, dec_embedding_size], -1, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    train_logits, infer_logits = decoding_layer(dec_embed_input, dec_embeddings,
                                                encoder_outputs, encoder_state, questions_vocab_size,
                                                sequence_length, rnn_size, num_layers,
                                                questions_vocab_to_int, keep_prob, batch_size)
    return train_logits, infer_logits


# --------------------------------------------------HYPER_PARAMETERS-----------------------------------------------------

epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75
min_line_length = 2
max_line_length = 24
# ------------------------------------------------DATA_PREPROCESSING-----------------------------------------------------

# Load the data
lines = open("movie-dialogs/movie_lines.txt", encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open("movie-dialogs/movie_conversations.txt", encoding='utf-8', errors='ignore').read().split('\n')

# Dictionary for id to line
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Conv sequence list
convs = []
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    convs.append(_line.split(","))

# Seperate the text into questions and answers
questions = []
answers = []

for conv in convs:
    for i in range(len(conv) - 1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i + 1]])

# Check current loading of data
limit = 0
for i in range(limit, limit + 5):
    print(questions[i])
    print(answers[i] + "\n")


# Text filter function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"we'd", "we would", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text


# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Print ssubset of clean answers
limit = 0
for i in range(limit, limit + 5):
    print(clean_questions[i])
    print(clean_answers[i] + "\n")

# Calculating the lengths of the sentences
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))

for answer in clean_answers:
    lengths.append(len(answer.split()))

# Create a dataframe so that the values can be inspected
lengths = pd.DataFrame(lengths, columns=['counts'])
print(lengths.describe())
print(np.percentile(lengths, 80))
print(np.percentile(lengths, 85))
print(np.percentile(lengths, 90))
print(np.percentile(lengths, 95))
print(np.percentile(lengths, 99))

# Filter out questions with long length
short_questions_temp = []
short_answers_temp = []
i = 0
for question in clean_questions:
    if len(question.split()) <= 24:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i += 1

i = 0
short_questions = []
short_answers = []
for answer in short_answers_temp:
    if len(answer.split()) <= 24:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

print("# of questions:", len(short_questions))
print("# of answers:", len(short_answers))
print("% of data used: {}%".format(round(len(short_questions) / len(questions), 4) * 100))

vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

threshold = 10
count = 0
for k, v in vocab.items():
    if v >= threshold:
        count += 1
print("Size of the total vocab : ", len(vocab))
print("Size of the vocab we will use : ", count)

# Provide a unique integer for each word

questions_vocab_to_int = {}
answers_vocab_to_int = {}

word_num = 0

for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        answers_vocab_to_int[word] = word_num
        word_num += 1

# Unique vocab tokens
codes = ['<PAD', '<EOS>', '<UNK>', '<GO>']

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int) + 1
    answers_vocab_to_int[code] = len(answers_vocab_to_int) + 1

questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

print("Checking the length of conversion dictionaries...")
print("questions_vocab_to_int: " + str(len(questions_vocab_to_int)))
print("questions_int_to_vocab: " + str(len(questions_int_to_vocab)))
print("answers_vocab_to_int: " + str(len(answers_vocab_to_int)))
print("answers_int_to_vocab: " + str(len(answers_int_to_vocab)))

# Add end of sentence token
for i in range(len(short_answers)):
    short_answers[i] += ' <EOS> '

questions_ints = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_ints.append(ints)

answer_ints = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answer_ints.append(ints)

print(len(questions_ints))
print(len(answer_ints))

# Calculate number of words replaced with <UNK>
word_count = 0
unk_count = 0

for question in questions_ints:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

for answer in answer_ints:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1
unk_ratio = round(unk_count / word_count, 4) * 100
print("Total number of words: ", word_count)
print("Total number of times <UNK> is used:", unk_count)
print("Percentage of words that are <UNK>:{}%".format(unk_ratio))

sorted_questions = []
sorted_answers = []
# sort questions by their length, fewer padding, faster training
for length in range(1, max_line_length+1):
    for i in enumerate(questions_ints):
        if len(i[1]) == length:
            sorted_questions.append(questions_ints[i[0]])
            sorted_answers.append(answer_ints[i[0]])

print(len(sorted_questions))
print(len(sorted_answers))
print()
for i in range(3):
    print(sorted_questions[i])
    print(sorted_answers[i])
    print()

# -----------------------------------------------------TRAINING----------------------------------------------------------
# Reset the graph
tf.reset_default_graph()

# Start the session
sess = tf.InteractiveSession()

input_data, targets, lr, keep_prob = model_inputs()
sequence_length = tf.placeholder_with_default(
    max_line_length,
    None,
    name='sequence_length'
)
input_shape = tf.shape(input_data)
# Training and inference logits
train_logits, inference_logits = seq2seq(tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length,
                                         len(answers_vocab_to_int), len(questions_vocab_to_int),
                                         encoding_embedding_size,
                                         decoding_embedding_size, rnn_size, num_layers, questions_vocab_to_int)
# Putting inference logits in a tensor
tf.identity(inference_logits, "logits")

with tf.name_scope("optimization"):
    # Loss Function
    cost = tf.contrib.seq2seq.sequence_loss(train_logits, targets, tf.ones(input_shape[0], sequence_length))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


def pad_sentence_batch(sentence_batch, vocab_to_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def batch_data(questions, answers, batch_size):
    for batch_i in range(0,len(questions)//batch_size):
        start_i = batch_i*batch_size
        question_batch = questions[start_i:start_i+batch_size]
        answer_batch = answers[start_i:start_i+batch_size]
        pad_question_batch = np.array(pad_sentence_batch(question_batch,questions_vocab_to_int))
        pad_answer_batch = np.array(pad_sentence_batch(answer_batch,answers_vocab_to_int))
        yield pad_question_batch,pad_answer_batch


# Validation split
train_valid_split = int(len(sorted_questions)*0.15)
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]
valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_answers[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))

display_step = 100 # Check training loss after every 100 batches
stop_early = 0
stop = 5 # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = ((len(train_questions))//batch_size//2)-1 # Modulus for checking validation loss
total_train_loss = 0 # Record the training loss for each display step
summary_valid_loss = [] # Record the validation loss for saving improvements in the model

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer)

for epoch_i in range(1,epochs+1):
    for batch_i, (questions_batch,answers_batch) in enumerate(batch_data(train_questions,train_answers,batch_size)):
        start_time = time.time()
        _,loss = sess.run([train_op,cost],{input_data: questions_batch, targets: answers_batch,
                                           lr: learning_rate,sequence_length: answers_batch.shape[1],
                                           keep_prob: keep_probability})
        total_train_loss += loss
        end_time = time.time()
        batch_time = start_time - end_time
        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs,
                          batch_i,
                          len(train_questions) // batch_size,
                          total_train_loss / display_step,
                          batch_time * display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: learning_rate,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!')
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.")
                stop_early += 1
                if stop_early == stop:
                    break

    if stop_early == stop:
        print("Stopping Training.")
        break


# In[44]:

def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''

    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]


# In[60]:

# Create your own input question
# input_question = 'How are you?'

# Use a question from the data as your input
random = np.random.choice(len(short_questions))
input_question = short_questions[random]

# Prepare the question
input_question = question_to_seq(input_question, questions_vocab_to_int)

# Pad the questions until it equals the max_line_length
input_question = input_question + [questions_vocab_to_int["<PAD>"]] * (max_line_length - len(input_question))
# Add empty questions so the the input_data is the correct shape
batch_shell = np.zeros((batch_size, max_line_length))
# Set the first question to be out input question
batch_shell[0] = input_question

# Run the model with the input question
answer_logits = sess.run(inference_logits, {input_data: batch_shell,
                                            keep_prob: 1.0})[0]

# Remove the padding from the Question and Answer
pad_q = questions_vocab_to_int["<PAD>"]
pad_a = answers_vocab_to_int["<PAD>"]

print('Question')
print('  Word Ids:      {}'.format([i for i in input_question if i != pad_q]))
print('  Input Words: {}'.format([questions_int_to_vocab[i] for i in input_question if i != pad_q]))

print('\nAnswer')
print('  Word Ids:      {}'.format([i for i in np.argmax(answer_logits, 1) if i != pad_a]))
print('  Response Words: {}'.format([answers_int_to_vocab[i] for i in np.argmax(answer_logits, 1) if i != pad_a]))
