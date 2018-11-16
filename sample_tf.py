import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

# %matplotlib inline
import matplotlib.pyplot as plt

import tempfile
logdir = tempfile.mkdtemp()

b = tf.Variable(tf.zeros((100,)))
W = tf.Variable(tf.random_uniform((784, 100),
                -1, 1))
x = tf.placeholder(tf.float32, (None, 784), name="x")
h_i = tf.nn.relu(tf.matmul(x, W) + b)
ops.reset_default_graph()
tf.global_variables_initializer()
x_batch = np.linspace(-1, 1, 101)
y_batch = x_batch * 2 + np.random.randn(*x_batch.shape) * 0.3
plt.scatter(x_batch, y_batch)
# plt.show()

x = tf.placeholder(tf.float32, shape=(None,), name="x")
y = tf.placeholder(tf.float32, shape=(None,), name="y")
w = tf.Variable(np.random.normal(), name="W")
y_pred = tf.mul(w, x)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# y_pred_batch = sess.run(y_pred, {x: x_batch})
# plt.figure(1)
# plt.scatter(x_batch, y_batch)
# plt.plot(x_batch, y_pred_batch)
# plt.show()

cost = tf.reduce_mean(tf.square(y_pred - y))
summary_op = tf.scalar_summary("cost", cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)
summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)
for t in range(30):
    cost_t, summary, _ = sess.run([cost, summary_op, train_op], {x: x_batch, y: y_batch})
    summary_writer.add_summary(summary, t)
    print(cost_t.mean())

# y_pred_batch = sess.run(y_pred, {x: x_batch})
# plt.figure(1)
# plt.scatter(x_batch, y_batch)
# plt.plot(x_batch, y_pred_batch)
# plt.show()

summary_writer.flush()





seq2seq = tf.nn.seq2seq
ops.reset_default_graph()
sess = tf.InteractiveSession()
seq_length = 5
batch_size = 64

vocab_size = 7
embedding_dim = 50

memory_dim = 100
enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32)
           for labels_t in labels]

# Decoder input: prepend some "GO" token and drop the final
# token of the encoder input
dec_inp = ([tf.zeros_like(enc_inp[0], dtype=np.int32, name="GO")]
           + enc_inp[:-1])

# Initial memory value for recurrence.
prev_mem = tf.zeros((batch_size, memory_dim))

cell = tf.nn.rnn_cell.GRUCell(memory_dim)

dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size, vocab_size,10)

loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)
tf.scalar_summary("loss", loss)
magnitude = tf.sqrt(tf.reduce_sum(tf.square(dec_memory[1])))
tf.scalar_summary("magnitude at t=1", magnitude)
summary_op = tf.merge_all_summaries()
learning_rate = 0.05
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)
logdir = tempfile.mkdtemp()
summary_writer = tf.train.SummaryWriter(logdir, sess.graph_def)

sess.run(tf.global_variables_initializer())


def train_batch(batch_size):
    X = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
         for _ in range(batch_size)]
    Y = X[:]

    # Dimshuffle to seq_len * batch_size
    X = np.array(X).T
    Y = np.array(Y).T

    feed_dict = {enc_inp[t]: X[t] for t in range(seq_length)}
    feed_dict.update({labels[t]: Y[t] for t in range(seq_length)})

    _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
    return loss_t, summary

for t in range(500):
    loss_t, summary = train_batch(batch_size)
    summary_writer.add_summary(summary, t)
summary_writer.flush()

X_batch = [np.random.choice(vocab_size, size=(seq_length,), replace=False)
           for _ in range(10)]
X_batch = np.array(X_batch).T

feed_dict = {enc_inp[t]: X_batch[t] for t in range(seq_length)}
dec_outputs_batch = sess.run(dec_outputs, feed_dict)
print(X_batch)
print([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch])