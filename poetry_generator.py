import numpy as np
import tensorflow as tf
import string
from sklearn.utils import shuffle

def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))

def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open('robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx

def idx2word(word2idx):
    idx2word = {i:w for w,i in list(word2idx.items())}
    return idx2word


class PoetryGenerator:
    
    def __init__(self, hidden_size, vocabulary_size, input_dims, session):
        self.H = hidden_size
        self.V = vocabulary_size
        self.D = input_dims
        V = self.V
        D = self.D
        H = self.H
        We = tf.random_normal((V, D))
        Wx = tf.random_normal((D, H))
        Wh = tf.random_normal((H, H))
        bh = tf.zeros((1, H))
        Wo = tf.random_normal((H, V))
        bo = tf.zeros((1, V))
        h0 = tf.zeros((H,))
        self.We = tf.Variable(We, name='We')
        self.Wx = tf.Variable(Wx, name='Wx')
        self.Wh = tf.Variable(Wh, name='Wh')
        self.bh = tf.Variable(bh, name='bh')
        self.Wo = tf.Variable(Wo, name='Wo')
        self.bo = tf.Variable(bo, name='bo')
        self.h0 = tf.Variable(h0, name='h0')
        self.tfX = tf.placeholder(tf.int32, (None,))
        self.tfy = tf.placeholder(tf.int32, (None,))
        self.save_mode = False
        self.saver = tf.train.Saver()
        self.sess = session
        
    def fit(self, X, epochs=2):
        N = len(X)
        wordvec = tf.nn.embedding_lookup(self.We, self.tfX)
        XWx = tf.matmul(wordvec, self.Wx)
        def recurrence(h_t1, XWx):
            h_t1 = tf.reshape(h_t1, (1, self.H))
            ht = tf.nn.relu(XWx + tf.matmul(h_t1, self.Wh) + self.bh)
            ht = tf.reshape(ht, (self.H,))
            return ht
        outputs = tf.scan(recurrence, XWx, self.h0)
        logits = tf.matmul(outputs, self.Wo) + self.bo
        self.predict = tf.nn.softmax(logits)
        labels = tf.squeeze(tf.one_hot(tf.reshape(self.tfy, (-1,1)), self.V), 1)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        train_op = tf.train.RMSPropOptimizer(0.001).minimize(cost)
        self.sess.run(tf.global_variables_initializer())
        if self.save_mode:
            self.saver.restore(self.sess, './rnn_model.ckpt')
        for epoch in range(epochs):
            X = shuffle(X)
            for i in range(N):
                inp = [0] + X[i]
                tar = X[i] + [1]
                _, costi = self.sess.run([train_op, cost], {self.tfX:inp, self.tfy:tar})
            print("Cost: ", costi, " Epoch: ", epoch+1)
            self.saver.save(self.sess, './rnn_model.ckpt')
            
    def generate(self, rand_begin, idx2, V):
    	sentence = [rand_begin]
    	inp = [rand_begin]
    	k = 0
    	#while k < 10 and sentence[-1] != 1:
    	while k < 10:
    		probs = self.sess.run(self.predict, {self.tfX:inp})
    		probs = probs[-1]
    		word = np.random.choice(V, p=probs)
    		sentence.append(word)
    		inp = sentence
    		k += 1
    	int_sentence = ""
    	for i in sentence:
    		if i != 1:
    			int_sentence += idx2[i] 
    			int_sentence += " "
    	return int_sentence
        
    def load(self):
        self.save_mode = True


sentences, word2idx = get_robert_frost()
idx2 = idx2word(word2idx)
model = PoetryGenerator(100, len(word2idx), 60, tf.InteractiveSession())
model.load()
model.fit(sentences, 1)
para = []
print("How many lines do you want to generate?")
N = input()
V = len(word2idx)
pi = np.zeros(V)
for sentence in sentences:
	pi[sentence[0]] += 1
pi /= pi.sum()
for i in range(int(N)):
	V = len(pi)
	X = np.random.choice(V, p=pi)
	k = model.generate(X, idx2, V)
	para.append(k)
for sentence in para:
	print(sentence)