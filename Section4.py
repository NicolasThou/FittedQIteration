import os
from PIL import Image
from save_simulation import *
import Section2 as s2
import Bilel


# Execute the simulation and saves images in a directory
if __name__ == "__main__":

    # delete all the pictures of previous simulation
    files = os.listdir('simulation/')
    for file in files:
        os.remove('simulation/' + file)

    x = s2.initial_state()
    i = 1
    while s2.is_final_state(x) is False:
        file = "simulation/simulation" + str(i).zfill(2) + ".png"

        # save the image in the 'simulation' folder
        save_caronthehill_image(x[0], 1, out_file=file)
        u = Bilel.forward_policy(x)
        x = s2.f(x, u)
        i += 1

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
