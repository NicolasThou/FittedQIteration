import os
import numpy as np
from PIL import Image
from save_simulation import *
import Section2 as s2


def create_images(i, p0, p1):
    step = 0.1

    # computes how much image transition we need
    nb = int(np.floor(np.abs(p1-p0)/step))

    # look the sens the car is moving
    if p0 < p1:
        sign = 1
    else:
        sign = -1

    file = "simulation/simulation" + str(i).zfill(2) + ".png"
    save_caronthehill_image(p0, 1, out_file=file)
    i += 1

    for n in range(nb):
        file = "simulation/simulation" + str(i).zfill(2) + ".png"
        # we add an image every step units
        save_caronthehill_image(p0 + sign*(n+1)*step, 1, out_file=file)
        i += 1

    return i


def visualize_policy(x, policy):
    # delete all the pictures of previous simulation
    files = os.listdir('simulation/')
    for file in files:
        os.remove('simulation/' + file)

    i = 1
    while not s2.is_final_state(x):
        u = policy(x)
        new_x = s2.f(x, u)
        i = create_images(i, x[0], new_x[0])
        x = new_x

    # create a GIF file of the simulation
    frames = []
    imgs = sorted(os.listdir('simulation/'))
    for name in imgs:
        name = "simulation/" + name
        img = Image.open(name)

        # add the image to the list of frames
        frames.append(img)

    # creates a GIF by concatenating all the frames
    frames[0].save('animated.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100)


# Execute the simulation and saves images in a directory
if __name__ == "__main__":
    x = s2.initial_state()
    visualize_policy(x, s2.random_policy)
