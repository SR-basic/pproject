'''
개발 초기 만들었던 아바타모드 실행 파일
주석처리된 부분들은 사용하지 않고, real_make_avatar를 통해 만들어진 이미지들을
주석처리되지 않은 부분들이 새로운 창을 띄워 보여준다.
'''




import cv2
import real_make_avatar as mk
import main

name = 'avatar'

# img1 = cv2.imread('./testimg/1.png', cv2.IMREAD_COLOR)
# img2 = cv2.imread('./testimg/2.png', cv2.IMREAD_COLOR)
# img3 = cv2.imread('./testimg/3.png', cv2.IMREAD_COLOR)

# img0_0 = cv2.imread('./merge_image/0_0.png', cv2.IMREAD_COLOR)
# img0_1 = cv2.imread('./merge_image/0_1.png', cv2.IMREAD_COLOR)
# img0_2 = cv2.imread('./merge_image/0_2.png', cv2.IMREAD_COLOR)
# img0_3 = cv2.imread('./merge_image/0_3.png', cv2.IMREAD_COLOR)
# img1_0 = cv2.imread('./merge_image/1_0.png', cv2.IMREAD_COLOR)
# img1_1 = cv2.imread('./merge_image/1_1.png', cv2.IMREAD_COLOR)
# img1_2 = cv2.imread('./merge_image/1_2.png', cv2.IMREAD_COLOR)
# img1_3 = cv2.imread('./merge_image/1_3.png', cv2.IMREAD_COLOR)
# img2_0 = cv2.imread('./merge_image/2_0.png', cv2.IMREAD_COLOR)
# img2_1 = cv2.imread('./merge_image/2_1.png', cv2.IMREAD_COLOR)
# img2_2 = cv2.imread('./merge_image/2_2.png', cv2.IMREAD_COLOR)
# img2_3 = cv2.imread('./merge_image/2_3.png', cv2.IMREAD_COLOR)

# 함수 이름 avatareye로 바꿔야할지도
# blink_animation 0=통상 상태(눈뜸), 1 = 눈 감음, 2 = 눈감음과 눈뜸 사이의 애니메이션
# mouth 0=입다뭄 , 1= 작은입, 2= 중간입, 3= 큰입
# detected_emotion ['Neutral','Happy','Sad','Angry','Surprise','None']
def show_avatar(blink_animation, mouth,detected_emotion,images):
    if detected_emotion == 5:
        detected_emotion = 0

    cv2.imshow(name,images[detected_emotion][blink_animation][mouth])

    if blink_animation == 2:
        cv2.waitKey(70)
    '''
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
   '''

        # cv2.imshow(name, img3)
        # cv2.waitKey(70)

    return 0