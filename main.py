import os
import pickle
from six.moves import urllib
import tflearn
from tflearn.data_utils import *
# download txt
path = "shakespeare_input.txt"
if not os.path.isfile(path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/shakespeare_input.txt", path)
# transform vector
max_len = 25
char_idx = None
char_idx_file = "char_idx.pickle"
# if not os.path.exists(char_idx_file):
#     os.mknod(char_idx_file)
if not os.path.isfile(char_idx_file):
    print("loading")
    char_idx = pickle.load(open(char_idx_file, 'rb'))
    # X,Y is sequence
    # char_idx is dictionary
X, Y, char_idx = textfile_to_semi_redundant_sequences(path,
                                                      seq_maxlen=max_len,
                                                        redun_step=3, pre_defined_char_idx=char_idx)
pickle.dump(char_idx, open(char_idx_file, "wb"))
print(len(char_idx))

# RNN
g = tflearn.input_data([None, max_len, len(char_idx)])
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)  # drop_rate = 0.5
g = tflearn.lstm(g, 512, return_seq=True)
g = tflearn.dropout(g, 0.5)
g = tflearn.lstm(g, 512)
g = tflearn.dropout(g, 0.5)
g = tflearn.fully_connected(g, len(char_idx), activation='softmax')  # len(char_idx) is length of softmax
g = tflearn.regression(g, optimizer="adam", loss='categorical_crossentropy', learning_rate=0.001)


# generate sequence
m = tflearn.SequenceGenerator(g, dictionary=char_idx, seq_maxlen=max_len, clip_gradients=5.0, checkpoint_path='model_shakespeare')
seed = random_sequence_from_textfile(path, max_len)


def train():
    # train
    epoch = 50
    for i in range(epoch):
        m.fit(X, Y, validation_set=0.1, batch_size=128, n_epoch=1, run_id="shakespeare")
        m.save("model/model_shakespeare")


def test():
    m.load("model/model_shakespeare")
    print("-----TESTING-----")
    print("--Test with temperature=1.0 --")
    print(m.generate(600, temperature=1.0, seq_seed=seed))
    print("--Test with temperature=0.5 --")
    print(m.generate(600, temperature=0.5, seq_seed=seed))


if __name__ == "__main__":
    #train()
    test()
