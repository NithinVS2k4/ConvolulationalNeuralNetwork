import numpy as np

from ConvolutionalNeuralNetworks import *
import pygame
from PIL import Image

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    MaxPooling(2),
    Convolutional((1, 13, 13), 4, 5),
    MaxPooling(2),
    Reshape((5, 5, 5), (5 * 5 * 5, 1)),
    Dense(5 * 5 * 5, 64),
    ReLU(),
    Dense(64, 2),
    Sigmoid()
]


def load_network_numpy(network, filename_prefix):
    for i, layer in enumerate(network):
        if isinstance(layer, Dense):
            layer.weights = np.load(f'{filename_prefix}_layer_{i}_weights.npy')
            layer.biases = np.load(f'{filename_prefix}_layer_{i}_biases.npy')
        if isinstance(layer, Convolutional):
            layer.kernels = np.load(f'{filename_prefix}_layer_{i}_kernels.npy')
            layer.biases = np.load(f'{filename_prefix}_layer_{i}_biases.npy')


if isinstance(network[0],Convolutional):
    prefix = "CNN"
else:
    prefix = "MLP"
load_network_numpy(network, f'NeuralNetworkModular/{prefix}/digit_recog_NN')

pygame.init()
scale = 20
screen = pygame.display.set_mode((28*scale + 200,28*scale))
color_locations = []

font = pygame.font.SysFont("Helvetica",30)

rect = pygame.Rect(0, 0, 28*scale, 28*scale)
sub = screen.subsurface(rect)

def input_from_image(image_path,size, log=False):
        image_file = Image.open(image_path)
        image_file = image_file.resize(size)
        image_file = image_file.convert('L')
        image_file.save('digit_image_resized.jpg')
        image = np.asarray(image_file)
        image = image/255
        if isinstance(network[0],Convolutional):
            image = image.reshape(1, 28, 28)
        else:
            image = image.flatten()
        return image

def get_prediction(network):
    x = input_from_image('digit_image.jpg', (28, 28))
    for layer in network:
        x = layer.forward(x)
    return x

def display_prediction(prediction):
    white = (255,255,255)
    grey = (125,125,125)
    density = 7
    pygame.draw.line(screen,white,(28*scale,0),(28*scale,28*scale))
    for i in range(1,density):
        pygame.draw.line(screen,grey,(28*scale,i*28*scale//density),(0,i*28*scale//density))
        pygame.draw.line(screen,grey,(i*28*scale//density,0),(i*28*scale//density,28*scale))
    y_offset = 80
    spacing = 40
    x_offset = 28*scale + 30
    pred =  np.argmax(prediction)
    for i in range(len(prediction)):
        if i == pred:
            color = (5, 173, 56)
        else:
            color = white
        screen.blit(font.render(f"{i} : {round(100*prediction[i][0],2)}%",True,color),(x_offset,y_offset+spacing*i))

clock = pygame.time.Clock()
running = True
frame = -1
while running:
    clock.tick(240)
    frame += 1
    screen.fill((0,0,0))

    for position in color_locations:
        pygame.draw.circle(screen,(255,255,255),position,scale)

    if frame%60 == 0:
        pygame.image.save(sub,'digit_image.jpg')
        prediction = get_prediction(network)
        frame = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                color_locations = []

    if pygame.mouse.get_pressed(num_buttons=3)[0]:
        mouse_pos = pygame.mouse.get_pos()
        if mouse_pos not in color_locations:
            color_locations.append(mouse_pos)
    pygame.draw.rect(screen,(0,0,0),pygame.Rect((scale*28,0,200,scale*28)))

    display_prediction(prediction)
    pygame.display.update()

pygame.quit()
