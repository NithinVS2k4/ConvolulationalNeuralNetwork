import matplotlib.pyplot as plt
from ConvolutionalNeuralNetworks import *
from MNIST_DataLoader import MnistDataloader
import time
import keras._tf_keras.keras.datasets.mnist as mnist

print("Loading...")

start_time = time.time()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Time taken to load dataset : {end_time-start_time} seconds")
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    AvgPooling(2),
    Convolutional((1, 13, 13), 4, 5),
    AvgPooling(2),
    Reshape((5, 5, 5), (5 * 5 * 5, 1)),
    Dense(5 * 5 * 5, 64),
    ReLU(),
    Dense(64, 2),
    Sigmoid()
]


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

x_train, y_train = preprocess_data(x_train, y_train, 100, ones_and_zeros = True)
x_test, y_test = preprocess_data(x_test, y_test, 8000, ones_and_zeros = True)


print((len(x_train),len(y_train),len(x_test),len(y_test)))


def load_network_numpy(network, filename_prefix):
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            layer.weights = np.load(f'{filename_prefix}_layer_{i}_weights.npy')
            layer.biases = np.load(f'{filename_prefix}_layer_{i}_biases.npy')
        if isinstance(layer, Convolutional):
            layer.kernels = np.load(f'{filename_prefix}_layer_{i}_kernels.npy')
            layer.biases = np.load(f'{filename_prefix}_layer_{i}_biases.npy')

def save_network_numpy(network, filename_prefix):
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            np.save(f'{filename_prefix}_layer_{i}_weights.npy', layer.weights)
            np.save(f'{filename_prefix}_layer_{i}_biases.npy', layer.biases)
        if isinstance(layer, Convolutional):
            np.save(f'{filename_prefix}_layer_{i}_kernels.npy', layer.kernels)
            np.save(f'{filename_prefix}_layer_{i}_biases.npy', layer.biases)


load = True
save = True
training = False
if load:
    if isinstance(network[0],Convolutional):
        prefix = "CNN"
    else:
        prefix = "MLP"
    load_network_numpy(network, f'NeuralNetworkModular/{prefix}/digit_recog_NN')
if training:
    error_list = train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=3,
    learning_rate=0.005,
    verbose=True)
    if save:
        if isinstance(network[0],Convolutional):
            prefix = "CNN"
        else:
            prefix = "MLP"
        save_network_numpy(network, f'NeuralNetworkModular/{prefix}/digit_recog_NN')

score = 0
error = 0
distribution = [0,0]
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    pred = np.argmax(output)
    true = np.argmax(y)
    if pred == true:
        score+=1
    distribution[pred] += 1

print(f"Accuracy : {100*score/len(x_test)}%")
print(f"Distribution : {distribution}")


plt.plot(error_list)
plt.show()
