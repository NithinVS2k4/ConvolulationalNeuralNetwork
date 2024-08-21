import numpy as np
import keras._tf_keras.keras.utils as utils
import time


# Dense layer
class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        limit_He = np.sqrt(2 / input_size)
        limit_Xavier = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit_He, limit_He, (output_size, input_size))
        self.biases = np.random.uniform(-limit_He, limit_He, (output_size,1))

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, learning_rate):
        # .reshape(-1,1) for making it a coloumn vector, .reshape(1,-1) for making it a row vector
        # .dot() for matrix multiplication
        weight_gradients = np.dot(output_gradient.reshape(-1, 1), self.input.reshape(1, -1))
        input_gradients = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weight_gradients
        self.biases -= learning_rate * output_gradient
        return input_gradients


class Convolutional:
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_depth = input_depth
        self.input_shape = input_shape
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.random(self.kernels_shape)
        self.biases = np.random.random(self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += cross_correlate(input[j], self.kernels[i, j], 'valid')
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        inputs_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = cross_correlate(self.input[j], output_gradient[i], 'valid')
                inputs_gradient[j] += convolve(output_gradient[i], self.kernels[i, j], 'full')

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return inputs_gradient


class MaxPooling():
    def __init__(self, filter_size):
        self.filter_size = filter_size
        self.filter_shape = (filter_size, filter_size)
        self.stride = filter_size

    def forward(self, input):
        self.input = np.asarray(input)
        input_depth, input_height, input_width = self.input.shape
        output_size = -(-input_height//self.filter_size)  #Ceiling division
        self.output = np.zeros((input_depth, output_size,output_size))
        self.max_array = np.zeros(self.input.shape)
        for d in range(input_depth):
          for i in range(output_size):
            for j in range(output_size):
              r_start = i*self.stride
              r_end = min(r_start + self.filter_size, input_height)
              c_start = j*self.stride
              c_end = min(c_start + self.filter_size, input_width)

              window = self.input[d, r_start:r_end, c_start:c_end]
              max = np.max(window)
              self.output[d, i, j] = max
              max_index = np.unravel_index(np.argmax(window), window.shape)
              self.max_array[d, r_start + max_index[0], c_start + max_index[1]] = 1

        return self.output

    def backward(self, output_gradient, dummy_parameter):
        input_gradient = np.zeros_like(self.input)
        for d in range(output_gradient.shape[0]):
            for i in range(self.output.shape[1]):
                for j in range(self.output.shape[2]):
                    r_start = i * self.stride
                    r_end = min(r_start + self.filter_size, self.input.shape[1])
                    c_start = j * self.stride
                    c_end = min(c_start + self.filter_size, self.input.shape[2])

                    max_window = self.max_array[d, r_start:r_end, c_start:c_end]
                    input_gradient[d, r_start:r_end, c_start:c_end] += max_window * output_gradient[d, i, j]

        return input_gradient


class AvgPooling:
    def __init__(self, filter_size):
        self.filter_size = filter_size
        self.filter_shape = (self.filter_size, self.filter_size)
        self.stride = self.filter_size

    def forward(self, input):
        self.input = np.asarray(input)
        input_depth, input_height, input_width = self.input.shape
        output_size = -(-input_height//self.filter_size)
        self.output = np.zeros((input_depth, output_size, output_size))

        for d in range(input_depth):
            for i in range(output_size):
                for j in range(output_size):
                    r_start = i*self.stride
                    r_end = min(r_start + self.filter_size, input_height)
                    c_start = j*self.stride
                    c_end = min(c_start + self.filter_size, input_width)

                    window = self.input[d, r_start:r_end, c_start:c_end]

                    self.output[d, i, j] = np.mean(window)

        return self.output

    def backward(self, output_gradient, dummy_parameter):
        input_gradient = np.zeros_like(self.input)
        for d in range(output_gradient.shape[0]):
            for i in range(output_gradient.shape[1]):
                for j in range(output_gradient.shape[2]):
                    r_start = i*self.stride
                    r_end = min(r_start + self.filter_size, self.input.shape[1])
                    c_start = j * self.stride
                    c_end = min(c_start + self.filter_size, self.input.shape[2])
                    window = self.input[d, r_start:r_end, c_start:c_end]
                    input_gradient[d,r_start:r_end, c_start : c_end] = window*output_gradient[d, i, j]/np.sum(window)

        return input_gradient

class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, dummy_parameter):
        return np.reshape(output_gradient, self.input_shape)


# Activations layers
class Activation():
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = np.asarray(input)
        return self.activation(input)

    def backward(self, output_gradient, dummy_parameter):
        return np.multiply(output_gradient, self.activation_prime(self.input))


class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_prime)


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)


class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: (x > 0).astype(float)
        super().__init__(relu, relu_prime)


def mse(y_true, y_pred):
    return np.sum(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true)

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def cross_entropy_prime(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

def cross_correlate(input, kernel, type='valid'):
    input = np.asarray(input)
    kernel = np.asarray(kernel)
    inp_size = input.shape[0]
    ker_size = kernel.shape[0]

    if type == 'valid':
        out_size = inp_size - ker_size + 1
        if out_size <= 0:
            raise ValueError("Invalid kernel size.")
        output = np.zeros((out_size, out_size))

        for i in range(out_size):
            for j in range(out_size):
                output[i, j] = np.einsum('ij,ij->', input[i:i + ker_size, j:j + ker_size], kernel)

    elif type == 'full':
        pad_size = ker_size - 1
        mod_input = np.pad(input, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=0)
        return cross_correlate(mod_input, kernel, 'valid')

    else:
        raise ValueError("Invalid type. Use 'valid' or 'full'.")

    return output


def convolve(input, kernel, type):
    rotated_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    return cross_correlate(input, rotated_kernel, type)


#Training helper functions


def preprocess_data(x, y, limit, ones_and_zeros):
    if ones_and_zeros:
        zero_index = np.where(y == 0)[0][:limit]
        one_index = np.where(y == 1)[0][:limit]
        all_indices = np.hstack((zero_index, one_index))
        all_indices = np.random.permutation(all_indices)
        x, y = x[all_indices], y[all_indices]
    else:
        x = x[:limit]
        y = y[:limit]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = utils.to_categorical(y)
    if ones_and_zeros:
        y = y.reshape(len(y), 2, 1)
    else:
        y = y.reshape(len(y), 10, 1)
    return x, y


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    error_list = []
    for e in range(epochs):
        error = 0
        i = 1
        start = time.time()
        for  x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
            if verbose and i %100==0:
                print(f"{i}/{len(x_train)} : {time.time()-start:.3f} seconds : {error/i:.4f} error")
                start = time.time()
            i+=1


        error /= len(x_train)
        error_list.append(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

    return error_list
