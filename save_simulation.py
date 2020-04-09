import pygame
import numpy as np
from math import atan2, degrees, pi, sqrt
import os
from PIL import Image
import domain


MAX_HEIGHT_SPEED = 100
WIDTH_SPEED = 30
MAX_SPEED = 3
MIN_SPEED = -3
LOC_WIDTH_FROM_BOTTOM = 20
LOC_HEIGHT_FROM_BOTTOM = 20
CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400
screen = None
car = None
pt = None
background = None
checked = False
size_car = None
width_car = None
height_car = None
size_pt = None
width_pt = None
height_pt = None
# Coloring
color_hill = pygame.Color(0, 0, 0, 0)
color_shill = pygame.Color(64, 163, 191, 0)
color_phill = pygame.Color(64, 191, 114, 0)
color_acc_line = pygame.Color(0, 0, 0, 0)

# Initialization of variables related to car on the hill
max_speed = 3
min_speed = -3
step_hill = 2.0 / CANVAS_WIDTH


def ppoints_to_angle(x1, x2):
    dx = x1[1] - x1[0]
    dy = x2[1] - x2[0]
    rads = atan2(-dy, dx)
    rads %= 2 * pi
    degs = degrees(rads)
    return degs


def rotate(image, rect, angle):
    """Rotate the image while keeping its center."""
    # Rotate the original image without modifying it.
    new_image = pygame.transform.rotate(image, angle)
    # Get a new rect with the center of the old rect.
    rect = new_image.get_rect(center=rect.center)
    return new_image, rect


def Hill(p):
    return p * p + p if p < 0 else p / (sqrt(1 + 5 * p * p))


def save_caronthehill_image(position, speed, out_file=None, close=False):
    global screen, car, pt, background, checked, size_pt, width_pt, height_pt, size_car, width_car, height_car
    if screen is None:
        screen = pygame.display.set_mode((CANVAS_WIDTH, CANVAS_HEIGHT))
        pygame.display.iconify()
    loc_width_from_bottom = 35
    loc_height_from_bottom = 70
    pt_pos1 = -0.5
    pt_pos2 = 0.5
    max_height_speed = 50
    width_speed = 30
    thickness_speed_line = 3

    # Image loading
    if car is None:
        car = pygame.image.load("car.png")
        car.convert_alpha()
        size_car = car.get_rect().size
        width_car = size_car[0]
        height_car = size_car[1]
    if pt is None:
        pt = pygame.image.load("pine_tree.png")
        pt.convert_alpha()
        size_pt = pt.get_rect().size
        width_pt = size_pt[0]
        height_pt = size_pt[1]

    # Surface loading
    surf = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    surf.convert()

    # Discretization of the hill function steps

    # Draw the background and the hill function altogether
    if not checked and not os.path.isfile("background_" + str(CANVAS_WIDTH) + "_" + str(CANVAS_HEIGHT) + ".png"):

        # hill function plot
        points = list(np.arange(-1, 1, step_hill))
        hl = list(map(Hill, points))
        range_h = range(CANVAS_HEIGHT)
        pix = 0
        for h in hl:
            x = pix
            y = ((CANVAS_HEIGHT) / 2) * (1 + h)

            y = int(round(y))
            for yo in range_h:
                if yo < y:
                    c = color_phill
                elif yo > y:
                    c = color_shill
                surf.set_at((x, CANVAS_HEIGHT - yo), c)

            surf.set_at((x, CANVAS_HEIGHT - y), color_hill)
            pix += 1
        pygame.image.save(surf, "background_" + str(CANVAS_WIDTH) + "_" + str(CANVAS_HEIGHT) + ".png")
        checked = True

    else:
        if background is None:
            background = pygame.image.load("background_" + str(CANVAS_WIDTH) + "_" + str(CANVAS_HEIGHT) + ".png")
        surf.blit(background, (0, 0))

    # Display pine trees
    surf.blit(pt, (round((CANVAS_WIDTH / 2) * (1 + pt_pos1)) - width_pt / 2,
                   CANVAS_HEIGHT - round(((CANVAS_HEIGHT) / 2) * (1 + Hill(pt_pos1))) - height_pt))
    surf.blit(pt, (round((CANVAS_WIDTH / 2) * (1 + pt_pos2)) - width_pt / 2,
                   CANVAS_HEIGHT - round(((CANVAS_HEIGHT) / 2) * (1 + Hill(pt_pos2))) - height_pt))

    # Display the car
    x_car = round((CANVAS_WIDTH / 2) * (1 + position)) - width_car / 2
    h_car = Hill(position)
    h_car_next = Hill(position + step_hill)
    y_car = CANVAS_HEIGHT - round(((CANVAS_HEIGHT) / 2) * (1 + h_car)) - height_car
    angle = ppoints_to_angle((position, position + step_hill), (h_car, h_car_next))
    rot_car, rect = rotate(car, pygame.Rect(x_car, y_car, width_car, height_car), 360 - angle)
    surf.blit(rot_car, rect)

    # Display car speed

    # Display black line
    rect = (CANVAS_WIDTH - loc_width_from_bottom - width_speed, CANVAS_HEIGHT - loc_height_from_bottom, width_speed,
            thickness_speed_line)
    surf.fill(color_acc_line, rect)

    pct_speed = abs(speed) / max_speed
    color_speed = (pct_speed * 255, (1 - pct_speed) * 255, 0)
    height_speed = max_height_speed * (pct_speed)

    loc_width = CANVAS_WIDTH - width_speed - loc_width_from_bottom
    loc_height = CANVAS_HEIGHT - loc_height_from_bottom + thickness_speed_line if speed < 0 else CANVAS_HEIGHT - loc_height_from_bottom - height_speed
    rect = (loc_width, loc_height, width_speed, height_speed)
    surf.fill(color_speed, rect)

    if out_file is not None:
        pygame.image.save(surf, out_file)
        return True
    else:
        return pygame.surfarray.pixels3d(surf)


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


def visualize_policy(x, policy, file_name):
    # delete all the pictures of previous simulation
    files = os.listdir('simulation/')
    for file in files:
        os.remove('simulation/' + file)

    i = 1
    while not domain.is_final_state(x):
        u = policy(x)
        new_x = domain.f(x, u)
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
    frames[0].save('{}.gif'.format(file_name), format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=100)



