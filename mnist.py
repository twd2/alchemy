import tensorflow as tf
import random

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
VERIFY_RATIO = 0.05
BATCH_SIZE = 50
ITERATIONS = 40000


def get_train_data(file):
    xs = []
    ys = []
    for line in file:
        tokens = line.strip().split(',')
        label = tokens[0]
        pixels = tokens[1:]
        if label == 'label':
            continue
        label = int(label)
        pixels = list(map(lambda p: float(p) / 255.0, pixels))
        ys.append([1 if i == label else 0 for i in range(10)])
        xs.append(pixels)
    return xs, ys


def get_test_data(file):
    xs = []
    for line in file:
        pixels = line.strip().split(',')
        if pixels[0] == 'pixel0':
            continue
        pixels = list(map(lambda p: float(p) / 255.0, pixels))
        xs.append(pixels)
    return xs


def write_submission(filename, ys):
    with open(filename, 'w') as f:
        f.write('ImageId,Label\n')
        for i in range(len(ys)):
            f.write('{},{}\n'.format(i + 1, ys[i]))


def make_weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, 0, 0.1))


def make_bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def main(_):
    # Create the model
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
        print(x)
        x_ = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        print(x_)
    with tf.name_scope('c1'):
        with tf.name_scope('weights'):
            w1 = make_weight_variable([5, 5, 1, 32])
        with tf.name_scope('bias'):
            b1 = make_bias_variable([32])
        c1 = tf.nn.relu(tf.nn.conv2d(x_, w1, [1, 1, 1, 1], 'SAME') + b1)
        print(c1)
    with tf.name_scope('s2'):
        s2 = tf.nn.max_pool(c1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        print(s2)
    with tf.name_scope('c3'):
        with tf.name_scope('weights'):
            w3 = make_weight_variable([5, 5, 32, 64])
        with tf.name_scope('bias'):
            b3 = make_bias_variable([64])
        c3 = tf.nn.relu(tf.nn.conv2d(s2, w3, [1, 1, 1, 1], 'VALID') + b3)
        print(c3)
    with tf.name_scope('s4'):
        s4 = tf.nn.max_pool(c3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        print(s4)
        s4_flat = tf.reshape(s4, [-1, 5 * 5 * 64])
        print(s4_flat)
    with tf.name_scope('c5'):
        with tf.name_scope('weights'):
            w5 = make_weight_variable([5 * 5 * 64, 1024])
        with tf.name_scope('bias'):
            b5 = make_bias_variable([1024])
        c5 = tf.nn.relu(tf.matmul(s4_flat, w5) + b5)
        print(c5)

    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32)

    # with tf.name_scope('c5_dropout'):
    #     c5_drop = tf.nn.dropout(c5, keep_prob)

    with tf.name_scope('f6'):
        with tf.name_scope('weights'):
            w6 = make_weight_variable([1024, 512])
        with tf.name_scope('bias'):
            b6 = make_bias_variable([512])
        f6 = tf.nn.relu(tf.matmul(c5, w6) + b6)
        print(f6)

    with tf.name_scope('f6_dropout'):
        f6_drop = tf.nn.dropout(f6, keep_prob)

    with tf.name_scope('output'):
        with tf.name_scope('weights'):
            w7 = make_weight_variable([512, 10])
        with tf.name_scope('bias'):
            b7 = make_bias_variable([10])
        y_pre = tf.matmul(f6_drop, w7) + b7
        y = tf.nn.softmax(y_pre)
        print(y)

    with tf.name_scope('output_expected'):
        y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pre))
    with tf.name_scope('train'):
        train_steps = [None, None]
        train_steps[0] = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        train_steps[1] = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # tf.summary.scalar('accuracy', accuracy)

    sess = tf.InteractiveSession()
    print(sess.graph)

    # init
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/log', sess.graph)
    tf.global_variables_initializer().run()

    print('Loading data...')
    with open('train.csv', 'r') as train_file:
       xs, ys = get_train_data(train_file)
    with open('test.csv', 'r') as train_file:
       xs_test = get_test_data(train_file)
    verify_count = int(len(xs) * VERIFY_RATIO)
    ids = list(range(len(xs)))
    random.shuffle(ids)
    xs = [xs[i] for i in ids]
    ys = [ys[i] for i in ids]
    xs_verify = xs[0:verify_count]
    ys_verify = ys[0:verify_count]
    xs = xs[verify_count:]
    ys = ys[verify_count:]
    print('Verify set {} records'.format(len(xs_verify)))
    print('Training set {} records'.format(len(xs)))
    print('Test set {} records'.format(len(xs_test)))

    print('Training...')
    for i in range(ITERATIONS):
        ids = random.sample(list(range(len(xs))), BATCH_SIZE)
        batch_xs = [xs[id] for id in ids]
        batch_ys = [ys[id] for id in ids]
        if (i + 1) % 100 == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print(i + 1, acc)
        train_step = train_steps[0 if i < ITERATIONS / 2 else 1]
        summary, _ = sess.run([merged, train_step],
                              feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        writer.add_summary(summary, i + 1)
        if (i + 1) % 1000 == 0:
            # verify trained model
            print('verify', sess.run(accuracy, feed_dict={x: xs_verify,
                                                          y_: ys_verify, keep_prob: 1.0}))
            argmax_y = tf.argmax(y, 1)
            argmax_ys = sess.run(argmax_y, feed_dict={x: xs_test, keep_prob: 1.0})
            write_submission('{}_submission.csv'.format(i + 1), argmax_ys)
    writer.close()

if __name__ == '__main__':
    tf.app.run(main=main)