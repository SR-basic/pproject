import cv2
import numpy as np
import img_reading as rd
from PIL import Image
from PIL import ImageChops
import time

'''
이 함수는 판떼기를 만드는 함수 
알파값을 이용한 이미지합성을 다룬다.
'''



visual_mode = True    # 이미지를 가져올때 전신 = False, 상반신 = True

main_head = rd.get_full_img_verPIL('./img/body/main_head.png', visual_mode)
main_body = rd.get_full_img_verPIL('./img/body/main_body.png', visual_mode)
cloth = rd.get_full_img_verPIL('./img/body/cloth.png', visual_mode)
# hair = rd.get_full_img_verPIL('./img/head_parts/hair.png',visual_mode)
back_hair = rd.get_full_img_verPIL('./img/head_parts/back_hair.png',visual_mode)
front_hair = rd.get_full_img_verPIL('./img/head_parts/front_hair.png',visual_mode)

class_labels=['Angry','Happy','Neutral','Sad','Surprise']

shift_x,shift_y = 0,10          #눈을 감았을떄 턱이 내려가는 애니메이션을 만들기위한 함수 y값만큼 턱이 내려간다.


# image_without_alpha = main_body[:,:,:3]                             # 전체
# bg = np.full(image_without_alpha.shape,(0,215,0), dtype=np.uint8)   # 크로마키 배경이미지

# 몸과 머리를 나누어 make_body,make_head의 함수로 나누었지만 실제 개발상 레이어의 순서도가 바뀌고 실행의 최적화를 위해
# 실제로 몸과 머리를 만들지는 않고 약간 혼합된 형태, 하지만 body가 back frame이고, hair가 front frame이라는 개념만 알아주면 될듯하다.
def make_body(main_body,back_hair=None,shoe=None,pants=None,cloth=None,another1=None,another2=None,main_head=None) :

    body = []
    for i in range(2) :

        bg = rd.make_bg(main_body)
        image_stack = [main_body]
        if back_hair :
            if i == 1:
                back_hair = shift_img(back_hair, shift_x,shift_y)
            image_stack.append(back_hair)
        if shoe :
            image_stack.append(shoe)
        if pants :
            image_stack.append(pants)
        if cloth :
            image_stack.append(cloth)
        if another1 :
            image_stack.append(another1)
        if another2 :
            image_stack.append(another2)
        if main_head :
            if i == 1:
                main_head = shift_img(main_head, shift_x,shift_y)
            image_stack.append(main_head)
        for image in image_stack :
            bg = Image.alpha_composite(bg, image)

        body.append(bg)

    return body


def make_hair(front_hair,another1 = None,another2=None) :
    image_stack=[]
    if front_hair :
        image_stack.append(front_hair)
    if another1 :
        image_stack.append(another1)
    if another2 :
        image_stack.append(another2)

    for image in image_stack :
        main_head = Image.alpha_composite(main_head, image)

    return main_head
'''
def make_body(layer_count) :
    image_stack = []
    for i in range(layer_count) :
        image_stack.append(check_exception(i))

    for i in range(image_stack) :
        main_body = Image.alpha_composite(main_body, main_head)
        '''

# for test
eyes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
mouth = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
def make_face(eyes123=[],mouth123=[],front_head = None) :
    face=[]
    emotion = []
    mouths = []
    i = 1
    for eye in range(len(eyes)) :
        for mou in range(len(mouth)) :
            if eye//3 != mou//4 :
                mou = (eye//3)*4+mou
            # 여기에 얼굴과 입 합성 함수
            # + 앞머리 합성 함수
            mouths.append(i)
            i += 1
            print("face[",eye//3,"][",eye%3,"][",mou%4,"] = ",eyes[eye]," + ",mouth[mou])
            if mou%4 == 3 :
                print("/n")
                face.append(mouths)
                mouths=[]
                break

        if eye%3 == 2 :
            emotion.append(face)
            face=[]


    for i in range(5) :
        print(emotion[i])

def shift_img(img,shift_x=0,shift_y=0) :
    shifted_img = ImageChops.offset(img,shift_x,shift_y)
    return shifted_img

def check_exception(index):
    # Checks if any of the layers have a known exception with another layer
    temp_file_path = index
    if is_exception(temp_file_path):
        check_exception(index)
    else:
        return temp_file_path

def is_exception(file_path):
    # Gets layer exceptions from file
    return False

def main():

    # start_time = time.perf_counter()


    body = make_body(main_body,back_hair=back_hair, cloth=cloth, main_head=main_head)
    for i in range(len(body)) :
        body[i].show()

    hair = make_hair(front_hair)
    emotion = make_face(front_head=hair)
    # 이러면 5*3*4의 앞얼굴 프레임이 생김

    # for문을 이용하여 body와 emotion을 하나씩 전부 합성
    # 3차원 배열을 유지하면서 body를 하나씩 합성하면 구현 쉬울듯

    # 그후 cv로 변환하든지, 사진으로 저장을 한번 시키던지(True,False로 저장여부를 뭍는게 좋을듯함)
    '''
    frame1 = Image.alpha_composite(body,head)

    shift_head = shift_img(head,shift_x,shift_y)
    frame2 = Image.alpha_composite(body,shift_head)

    # frame1.show()
    frame2.show()
    '''

    # elapsed = (time.perf_counter() - start_time) * 1000
    # print ( "수행시간 = %0.2f ms" % elapsed)
    # cv2.waitKey(0) & 0xFF == 27

if __name__ == "__main__":
    main()
