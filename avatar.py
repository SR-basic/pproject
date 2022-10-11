import cv2

name = 'avatar'

# img1 = cv2.imread('./testimg/1.png', cv2.IMREAD_COLOR)
# img2 = cv2.imread('./testimg/2.png', cv2.IMREAD_COLOR)
# img3 = cv2.imread('./testimg/3.png', cv2.IMREAD_COLOR)

img0_0 = cv2.imread('./merge_image/0_0.png', cv2.IMREAD_COLOR)
img0_1 = cv2.imread('./merge_image/0_1.png', cv2.IMREAD_COLOR)
img0_2 = cv2.imread('./merge_image/0_2.png', cv2.IMREAD_COLOR)
img0_3 = cv2.imread('./merge_image/0_3.png', cv2.IMREAD_COLOR)
img1_0 = cv2.imread('./merge_image/1_0.png', cv2.IMREAD_COLOR)
img1_1 = cv2.imread('./merge_image/1_1.png', cv2.IMREAD_COLOR)
img1_2 = cv2.imread('./merge_image/1_2.png', cv2.IMREAD_COLOR)
img1_3 = cv2.imread('./merge_image/1_3.png', cv2.IMREAD_COLOR)
img2_0 = cv2.imread('./merge_image/2_0.png', cv2.IMREAD_COLOR)
img2_1 = cv2.imread('./merge_image/2_1.png', cv2.IMREAD_COLOR)
img2_2 = cv2.imread('./merge_image/2_2.png', cv2.IMREAD_COLOR)
img2_3 = cv2.imread('./merge_image/2_3.png', cv2.IMREAD_COLOR)

# 함수 이름 avatareye로 바꿔야할지도
# 0=통상 상태(눈뜸), 1 = 눈 감음, 2 = 눈감음과 눈뜸 사이의 애니메이션
def show_avatar(blink_animation = 0, mouth = 0):
    if blink_animation == 0 :
        if mouth == 0 :
            img = cv2.resize(img0_0, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
        elif mouth == 1 :
            img = cv2.resize(img0_1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
        elif mouth == 2 :
            img = cv2.resize(img0_2, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
        elif mouth == 3 :
            img = cv2.resize(img0_3, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
    elif blink_animation == 1 :
        if mouth == 0:
            img = cv2.resize(img1_0, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
        elif mouth == 1:
            img = cv2.resize(img1_1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
        elif mouth == 2:
            img = cv2.resize(img1_2, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
        elif mouth == 3:
            img = cv2.resize(img1_3, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
    elif blink_animation == 2 :
        if mouth == 0:
            img = cv2.resize(img2_0, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
            cv2.waitKey(70)
        elif mouth == 1:
            img = cv2.resize(img2_1, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
            cv2.waitKey(70)
        elif mouth == 2:
            img = cv2.resize(img2_2, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
            cv2.waitKey(70)
        elif mouth == 3:
            img = cv2.resize(img2_3, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_CUBIC)
            cv2.imshow(name, img)
            cv2.waitKey(70)
        # cv2.imshow(name, img3)
        # cv2.waitKey(70)

    return 0