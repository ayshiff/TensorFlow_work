import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

rand = random.choice(mnist.test.images)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
s = ax.imshow(rand.reshape((28,28)), cmap = matplotlib.cm.binary)
plt.show()


x = tf.placeholder(tf.float32, [None, 28*28])
classes = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([28*28, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=classes, logits=y))

entrainement = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(entrainement, feed_dict={x: batch_xs, classes: batch_ys})

# Test trained model
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(classes, 1))
performance = tf.reduce_mean(tf.cast(prediction, tf.float32))
taux_reussite = sess.run(performance, feed_dict={x: mnist.test.images,
                                  classes: mnist.test.labels})


img = rand.reshape(1, 784)
prediction=tf.argmax(y,1)

print (prediction.eval(feed_dict={x: img}))
