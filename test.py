import tensorflow as tf
import tflearn
from tflearn.data_utils import *
# with tf.Session() as sess:
#     saver = tf.train.Saver(tf.global_variables_initializer)
#     saver.restore(sess, tf.train.latest_checkpoint("model"))
#     print(m.generate(600, temperature=1.0, seq_seed=seed))
# max_len = 25
# char_idx = None
# path = "shakespeare_input.txt"
# seed = random_sequence_from_textfile(path, max_len)
# g = tflearn.input_data([None, max_len, len(char_idx)])
# g = tflearn.lstm(g, 512, return_seq=True)
# g = tflearn.dropout(g, 0.5)  # drop_rate = 0.5
# g = tflearn.lstm(g, 512, return_seq=True)
# g = tflearn.dropout(g, 0.5)
# g = tflearn.lstm(g, 512)
# g = tflearn.dropout(g, 0.5)
# g = tflearn.fully_connected(g, len(char_idx), activation='softmax')  # len(char_idx) is length of softmax
# g = tflearn.regression(g, optimizer="adam", loss='categorical_crossentropy', learning_rate=0.001)
# g.load("model/model_shakespeare-10719")
# print("-----TESTING-----")
# print("--Test with temperature=1.0 --")
# print(g.generate(600, temperature=1.0, seq_seed=seed))
max_len = 25
path = "shakespeare_input.txt"
seed = random_sequence_from_textfile(path, max_len)
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables_initializer)
    new_saver = tf.train.import_meta_graph("model/model_shakespeare-10719.meta")
    saver.restore(sess, tf.train.latest_checkpoint("model"))
    print("-----TESTING-----")
    print("--Test with temperature=1.0 --")
    print(new_saver.generate(600, temperature=1.0, seq_seed=seed))