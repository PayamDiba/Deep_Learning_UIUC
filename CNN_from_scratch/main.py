from conv2d_NN import ConvNN_classifier
import h5py
import numpy as np


MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] ).reshape((len(MNIST_data['x_train']), 28, 28))
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] ).reshape((len(MNIST_data['x_test']), 28, 28))
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()


model = ConvNN_classifier(input_size = (28,28), number_classes = 10)
model.build_graph(size = (3,3,4), activation_hidden = 'sigmoid')
model.train(x_train, y_train, x_test, y_test, number_epoches = 10, print_performance = True)
