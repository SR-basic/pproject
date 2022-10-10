import cv2

name = 'avatar'

img1 = cv2.imread('./testimg/1.png', cv2.IMREAD_COLOR)
img2 = cv2.imread('./testimg/2.png', cv2.IMREAD_COLOR)
img3 = cv2.imread('./testimg/3.png', cv2.IMREAD_COLOR)

# 함수 이름 avatareye로 바꿔야할지도
# 0=통상 상태(눈뜸), 1 = 눈 감음, 2 = 눈감음과 눈뜸 사이의 애니메이션
def show_avatar(blink_animation = 0):
    if blink_animation == 0 :
        cv2.imshow(name, img1)
    elif blink_animation == 1 :
        cv2.imshow(name, img2)
    elif blink_animation == 2 :
        cv2.imshow(name, img3)
        cv2.waitKey(70)

    return 0