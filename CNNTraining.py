import matplotlib.pyplot as plt
from NeuralNetworks import *
from MNIST_DataLoader import MnistDataloader
import time
import keras._tf_keras.keras.datasets.mnist as mnist


input_path = ''
training_images_filepath = 'MNIST_HandwrittenDigits/train-images-idx3-ubyte/train-images-idx3-ubyte'
training_labels_filepath = 'MNIST_HandwrittenDigits/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
test_images_filepath = 'MNIST_HandwrittenDigits/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
test_labels_filepath = 'MNIST_HandwrittenDigits/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

print("Loading...")

start_time = time.time()

load_npy_dataset = False
if not load_npy_dataset:
    # mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    #
    # train_and_save = True
    #
    # data_set = mnist_dataloader.load_data(load_train = train_and_save,augment=False, replace_set=True, num_of_copies=3)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
else:
    #Augmented data with 3 180,000 training images
    x_train = np.load('MNIST_HandwrittenDigits/x_train.npy')
    y_train = np.load('MNIST_HandwrittenDigits/y_train.npy')
    x_test = np.load('MNIST_HandwrittenDigits/x_test.npy')
    y_test = np.load('MNIST_HandwrittenDigits/y_test.npy')
end_time = time.time()
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
