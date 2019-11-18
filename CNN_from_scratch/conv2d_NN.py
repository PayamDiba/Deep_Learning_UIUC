import numpy as np
from functions import relu, d_relu, sigmoid, d_sigmoid, softmax

class ConvNN_classifier(object):
    def __init__(self, input_size, number_classes):
        """
        input_size = (nRow * nCol)
        """
        self.input_size_ = input_size
        self.num_class_ = number_classes
        self.params = {}

        self.activation_dict = { 'relu': (relu, d_relu),
                            'sigmoid': (sigmoid, d_sigmoid),
                            }


    def build_graph(self, size, activation_hidden):
        """
        This builds a 2d convolutional neural network with given size.

        Inputs:
        size = tuple(size_y, size_x, n_channel) where size_y and size_x are size of each filter
        in Y and X directions. n_channel is the number of filters


        Free parameters are initilized
        for input data x:

        z = x*K  ### |K| = (Sy,Sx,nChannel)  ;  z = (nData, nRow - Sy + 1, nCol - Sx + 1, nChannel)
        H = sigma(z)   ###    |H| = (nData, nRow - Sy + 1, nCol - Sx + 1, nChannel)
        U = W*.H + b  ### |W| = (nRow - Sy + 1, nCol - Sx + 1, nChannel, nClasses), |b| = nClasses * 1
        y^ = softmax(U)    ####   |y^| = nData * K
        """
        self.nChannel_ = size[-1]
        self.conv_size_ = size

        self.params['K'] = np.true_divide(1, np.sqrt(size[0] * size[1])) * np.random.normal(loc = 0.0, scale = 1.0, size = size)

        fc_size_y = self.input_size_[0] - size[0] + 1
        fc_size_x = self.input_size_[1] - size[1] + 1
        self.params['W'] = np.true_divide(1, np.sqrt(fc_size_y * fc_size_x)) * np.random.normal(loc = 0.0, scale = 1.0, size = (fc_size_y, fc_size_x, size[2], self.num_class_))
        self.params['b'] = np.true_divide(1, np.sqrt(self.num_class_)) * np.random.normal(loc = 0.0, scale = 1.0, size = (self.num_class_,))

        self.hidden_ = 0 #(nData, nRow - Sy + 1, nCol - Sx + 1, nChannel)
        self.z_ = 0 #(nData, nRow - Sy + 1, nCol - Sx + 1, nChannel)

        #TODO:
        self.activation = self.activation_dict[activation_hidden][0]
        self.d_activation = self.activation_dict[activation_hidden][1]

    def segmentize_(self, x, filter_size):
        """
        Segmentize the input x into a tensor suitable for convolution with filter of size filter_size:

        x: numpy.array(nData, height, width)
        filter_size: tuple(Sy,Sx)

        returns: numpy.array(nData, height - Sy + 1, width - Sx + 1, Sy, Sx)
        """
        h = np.shape(x[0])[0]
        w = np.shape(x[0])[1]

        Sy = filter_size[0]
        Sx = filter_size[1]

        ret = []
        for currX in x:
            currSeg = []
            for r in range(h - Sy + 1):
                seg_r = []
                for c in range(w - Sx + 1):
                    seg_r.append(currX[r:r+Sy,c:c+Sx])
                currSeg.append(seg_r)
            ret.append(currSeg)

        return np.array(ret)

    def convolution(self, conv_filter, X):
        """
        To calculate the convolution of a given all filters and data

        conv_filter = np.array(Sy * Sx, nChannel)
        X: np.array(nData, height * width)

        returns np.array(nData, height - Sy + 1, width - Sx + 1, nChannel)
        """

        x_seg = self.segmentize_(X, np.shape(conv_filter))

        return np.tensordot(x_seg, conv_filter, axes = 2)

    def feed_forward(self, x):

        """
        x: Input, numpy array (nData * input_size)
        returns numpy array (nData * K)
        """
        #TODO:Take care of dimension x: nData * height * width
        if x.ndim == 2:
            x = np.expand_dims(x, axis = 0)
            nData = 1
        else:
            nData = x.shape[0]

        z = self.convolution(self.params['K'], x)
        self.z_ = np.copy(z)

        H = self.activation(z)
        self.hidden_ = np.copy(H) #shape(H) = (nData, nRow - Sy + 1, nCol - Sx + 1, nChannel)

        b = np.tile(self.params['b'], (nData,1))
        U = np.tensordot(H,self.params['W'],3) + b
        y_hat = [softmax(u) for u in U]

        return np.array(y_hat)

        # softmax_U = []
        # for currH in H:
        #     currU = []
        #     for k in range(self.num_class_):
        #         u_k = 0
        #         for h in range(self.nChannel_):
        #             u_k += self.convolution(self.params['W'][:,:,h, k], currH[:,:,h])[0,0]
        #             #print (self.params['W'][:,:,h, k].shape)
        #             #print (currH[:,:,h].shape)
        #
        #         currU.append(u_k)
        #     softmax_U.append(softmax(currU))
        #
        #
        # softmax_U = np.array(softmax_U) #shape(U) = nData * num_class_
        # return softmax_U

    def hot_encode_(self, classes):
        """
        classes: numpy array (n,):
        returns: numpy array(n, nClasses)
        """
        if np.shape(classes) == ():
            nData = 1
        else:
            nData = len(classes)

        ret = np.zeros((nData, self.num_class_))

        ret[np.arange(nData), classes] = 1

        return ret

    def back_propagate(self, x, y_true):

        """
        For now this function assumes that nData = 1. This is suitable for SGD


        x: Input, numpy array of input_size
        y_true: scalar

        returns dLoss/dK (Sy,Sx,nChannel), dLoss/dW (nRow - Sy + 1, nCol - Sx + 1, nChannel, nClasses), dLoss/db (num_class_,)
        """
        y_hat = self.feed_forward(x)[0] # 1 * K
        y_true_encode = self.hot_encode_(y_true) # 1* K

        dU = y_hat - y_true_encode # 1*K
        #print(y_hat.shape)

        db = np.copy(dU) # 1*K

        H = np.expand_dims(self.hidden_[0], axis = 3)
        dW = np.matmul(H, dU) # nRow - Sy + 1, nCol - Sx + 1, nChannel, nClasses

        delta = np.matmul(self.params['W'], np.transpose(dU)) # nRow - Sy + 1, nCol - Sx + 1, nChannel
        delta = np.squeeze(delta)
        #print np.shape(delta)


        CF = np.multiply(delta, self.d_activation(self.z_[0]))
        dK = self.convolution(CF, np.expand_dims(x, axis = 0))[0]


        return dK, dW, db.flatten()

    def Loss (self, x, y_true):
        """
        Calculates averaged cross entropy loss for multi-class classification


        x: Input, numpy array (nData * df)
        y_true: numpy array (nData,)
        """
        nData = len(x)

        y_true_encode = self.hot_encode_(y_true)
        y_hat = self.feed_forward(x)

        ret = np.multiply(y_true_encode, y_hat)
        return np.true_divide(np.sum(ret),nData)

    def update_params(self, learn_rate, x, y_true):

        """
        Updates free parameters according the gradient descent. Gradients are calculated
        using the input x and y_true

        x: np.array (input_size) # nData = 1
        y_true: scalar
        """

        dK, dW, db = self.back_propagate(x, y_true)

        self.params['K'] -= learn_rate * dK
        self.params['W'] -= learn_rate * dW
        self.params['b'] -= learn_rate * db


    def learning_rate(self, epochs):
        if epochs < 5:
            return 0.01
        elif (epochs < 10):
            return 0.001
        elif (epochs < 15):
            return 0.0001
        else:
            return 0.00001


    def train(self, x_train, y_train, x_test, y_test, number_epoches, print_performance = True):
        """
        This trains neural network using stochastic gradient descent

        It also makes sure that the whole training data is used in each epoch
        if print_performance = True, averaged performance on train and test data is reported after each epoch
        """

        ids = np.arange(len(x_train))

        for nE in range(number_epoches):
            np.random.shuffle(ids)
            LR = self.learning_rate(nE)

            for i in ids:
                currX = x_train[i]
                currY = y_train[i]

                self.update_params(LR, currX, currY)

            if print_performance:

                test_loss = self.feed_forward(x_test)
                test_pred = np.argmax(test_loss, axis = 1)
                eval_test = test_pred == y_test
                num_true_test = np.sum(eval_test)
                perf_test = np.true_divide(num_true_test, len(x_test))

                print ("Epoch: " + str(nE+1))
                print ("Test Performance: " + str(perf_test))
