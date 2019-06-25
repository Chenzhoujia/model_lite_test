import tensorflow as tf
import argparse
import cv2
from tqdm import tqdm
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
# Architecture
hidden_size = 32
input_size = 784
image_width = 28

# Other
print_interval = 200
random_seed = 123

def load_graph(frozen_graph_filename):
    # 加载protobug文件，并反序列化成graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph_:
        # 将读出来的graph_def导入到当前的Graph中
        # 为了避免多个图之间的明明冲突，增加一个前缀
        tf.import_graph_def(graph_def, name="prefix")

    return graph_


if __name__ == '__main__':

    frozen_model_filename = "./ae-conv.pb"
    # 从pb文件中读取图结构
    graph = load_graph(frozen_model_filename)

    # 列举所有的操作
    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('prefix/input:0')
    y = graph.get_tensor_by_name('prefix/decoding:0')

    mnist = input_data.read_data_sets("./", validation_size=0)

    ##########################
    ### VISUALIZATION
    ##########################

    n_images = 15

    fig, axes = plt.subplots(nrows=2, ncols=n_images,
                             sharex=True, sharey=True, figsize=(20, 2.5))
    test_images = mnist.test.images[:n_images]
    np.savetxt("./test_image.txt", test_images, fmt='%f', delimiter=',')
    test_images = np.loadtxt('./test_image.txt', delimiter=',')
    # We launch a Session
    with tf.Session(graph=graph) as sess:
        # 不用执行初始化，因为全是常量
        decoded = sess.run(y, feed_dict={x: test_images})

for i in range(n_images):
    for ax, img in zip(axes, [test_images, decoded]):
        ax[i].imshow(img[i].reshape((image_width, image_width)), cmap='binary')

plt.savefig("result_ae-conv-pb.png")