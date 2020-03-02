import os
from PIL import Image
from save_simulation import *


# Execute the simulation and saves images in a directory
if __name__ == "__main__":

    # delete all the pictures of previous simulation
    files = os.listdir('simulation/')
    for f in files:
        os.remove('simulation/' + f)

    # simulates the car going from left to right of the hill
    for i in range(100):
        file = "simulation/simulation" + str(i).zfill(2) + ".png"
        position = -1 + 2*(i/100)

        # save the image in the 'simulation' folder
        save_caronthehill_image(position, 1, out_file=file)

    # create a GIF file of the simulation
    frames = []
    imgs = sorted(os.listdir('simulation/'))
    print(imgs)
    for name in imgs:  # TODO problem, name is not selected in the same order of the images 00, 01, ..., 99
        name = "simulation/" + name
        #print(name)
        img = Image.open(name)

        # add the image to the list of frames
        frames.append(img)

    # creates a GIF by concatenating all the frames
    frames[0].save('animated.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=0, loop=0)
