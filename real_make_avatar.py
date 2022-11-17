import cv2
import numpy as np
import img_reading as rd
from PIL import Image
from PIL import ImageChops
import glob
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

pre_eyes= glob.glob('./img/head_parts/eye/*.png')
eyes = []
for i in pre_eyes:
    img = rd.get_full_img_verPIL(i)
    eyes.append(img)

pre_mouth= glob.glob('./img/head_parts/mouth/*.png')
mouth = []
for i in pre_mouth:
    img = rd.get_full_img_verPIL(i)
    mouth.append(img)

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
        main_head = Image.alpha_composite(front_hair, image)

    return main_head
'''
def make_body(layer_count) :
    image_stack = []
    for i in range(layer_count) :
        image_stack.append(check_exception(i))

    for i in range(image_stack) :
        main_body = Image.alpha_composite(main_body, main_head)
        '''

def make_face(front_hair) :
    emotion = []
    face=[]
    mouths = []
    for eye in range(len(eyes)) :
        for mou in range(len(mouth)) :
            if eye//3 != mou//4 :
                mou = (eye//3)*4+mou
            img = Image.alpha_composite(eyes[eye],mouth[mou])
            img = Image.alpha_composite(front_hair,img)
            mouths.append(img)
            if mou%4 == 3 :
                face.append(mouths)
                mouths=[]
                break

        if eye%3 == 2 :
            emotion.append(face)
            face=[]

    return emotion



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

    # 배경과 뒷머리를 포함한 몸통을 +기본 머리를 만듭니다.
    body = make_body(main_body,back_hair=back_hair, cloth=cloth, main_head=main_head)
    # 머리카락을 머리카락 데코레이션과 함께 만듭니다.
    hair = make_hair(front_hair)
    # hair.show()
    # 불러운 입 배열과 눈 배열을 각각 합성하고, 앞에 머리카락을 붙힙니다.
    emotion = make_face(front_hair=hair)

    # body와 emotion을 합칩니다.
    # i = [기본,기쁨,슬픔,화남,놀람] j = [눈 뜸,눈 감음,눈 애니메이션프레임] l = [다문입,작은입,중간입,큰입]
    # for문을 통해 3차원 배열을 전부 돌면서 모든 경우의 표정을 body와 결합시킵니다.
    cv_emotion = [[[0 for _ in range(len(emotion[0][0]))] for _ in range(len(emotion[0]))] for _ in range(len(emotion))]
    for i in range(len(emotion)) :
        for j in range(len(emotion[0])) :
            for l in range(len(emotion[0][0])) :
                if j == 1 :
                    # j가 1일때, 즉 눈이 감겼을떄 나오는 애니메이션을 표현하기위해 body는 머리가 아래로 내려간 이미지,
                    # emotion는 아래로내려간 후처리가 되어있지 않으므로 이 연산과정에서 아래로 내린다.
                    emotion[i][j][l] = shift_img(emotion[i][j][l], shift_x,shift_y)
                    emotion[i][j][l] = Image.alpha_composite(body[1], emotion[i][j][l])
                else :
                    emotion[i][j][l] = Image.alpha_composite(body[0], emotion[i][j][l])
                # if i == 2 :
                #만약 연산된 이미지 보고싶다면 아래 show주석 해제,위에있는 i ==2는 슬픔얼굴을 보여주겠죠
                #   emotion[i][j][l].show()
                cv_emotion[i][j][l] = rd.convert_PIL_to_CV(emotion[i][j][l])
                # cv2.imshow('test',cv_emotion[i][j][l])
                # cv2.waitKey(1000)

    return cv_emotion
    # 요리가 완료된 변수는 emotion
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
