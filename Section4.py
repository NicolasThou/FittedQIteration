import os
from PIL import Image
from save_simulation import *
import section1 as s1


# Execute the simulation and saves images in a directory
if __name__ == "__main__":

    # delete all the pictures of previous simulation
    files = os.listdir('simulation/')
    for file in files:
        os.remove('simulation/' + file)

    x = s1.initial_state()
    i = 1
    while s1.is_final_state(x) is False:
        file = "simulation/simulation" + str(i).zfill(2) + ".png"

        # save the image in the 'simulation' folder
        save_caronthehill_image(x[0], 1, out_file=file)
        u = s1.random_policy(x)
        x = s1.f(x, u)
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
                   duration=0, loop=0)
