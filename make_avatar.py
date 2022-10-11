
'''
import cv2
import numpy as np

reszie_px = 0.3

# main
main_body = cv2.imread('./img/body/main_body.png', cv2.IMREAD_UNCHANGED)
main_head = cv2.imread('./img/body/main_head.png', cv2.IMREAD_UNCHANGED)

# body_parts
cloth1 = cv2.imread('./img/body/cloth.png', cv2.IMREAD_UNCHANGED)
# cloth2 = cv2.imread('',cv2.IMREAD_COLOR)

# head_parts
hair = cv2.imread('./img/head_parts/hair.png', cv2.IMREAD_UNCHANGED)

eye0 = cv2.imread('./img/head_parts/eye_0.png', cv2.IMREAD_UNCHANGED)
eye1 = cv2.imread('./img/head_parts/eye_1.png', cv2.IMREAD_UNCHANGED)
eye2 = cv2.imread('./img/head_parts/eye_2.png', cv2.IMREAD_UNCHANGED)

mouth0 = cv2.imread('./img/head_parts/mouth_0.png', cv2.IMREAD_UNCHANGED)
mouth1 = cv2.imread('./img/head_parts/mouth_1.png', cv2.IMREAD_UNCHANGED)
mouth2 = cv2.imread('./img/head_parts/mouth_2.png', cv2.IMREAD_UNCHANGED)
mouth3 = cv2.imread('./img/head_parts/mouth_3.png', cv2.IMREAD_UNCHANGED)


def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

def make_head(main_head,hair) :

    head = add_transparent_image(main_head,hair)
    head = cv2.resize(head, None, fx=reszie_px, fy=reszie_px, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('name', head)
    return 0

def make_body() :

    return 0

def main():
    not_bg = main_body.copy()
    bg = np.zeros(not_bg, np.uint8)
    bg_body = add_transparent_image(bg ,main_body)
    make_head(bg_body,cloth1)

    make_head()

    background = main_head
    overlay = hair
    x_offset = 0
    y_offset = 0
    print("arrow keys to move the dice. ESC to quit")
    while True:
        img = background.copy()
        add_transparent_image(img, overlay, x_offset, y_offset)
        img = cv2.resize(img, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)

        cv2.imshow("", img)
        print("1")
        key = cv2.waitKey()
        if key == 87:
            y_offset -= 10
            print("0") # up
        if key == 83:
            y_offset += 10
            print("1")# down
        if key == 65:
            x_offset -= 10
            print("2")# left
        if key == 68:
            x_offset += 10
            print("3")# right
        if key == 27: break  # escape


    # make_head()
    # cv2.waitKey()
    return 0

if __name__ == "__main__":
    main()

'''