import cv2
import numpy as np
import img_reading as rd
from PIL import Image
from PIL import ImageChops
import time

visual_mode = True    # 이미지를 가져올때 전신 = False, 상반신 = True

main_head = rd.get_full_img_verPIL('./img/body/main_head.png', visual_mode)
main_body = rd.get_full_img_verPIL('./img/body/main_body.png', visual_mode)
cloth = rd.get_full_img_verPIL('./img/body/cloth.png', visual_mode)
hair = rd.get_full_img_verPIL('./img/head_parts/hair.png',visual_mode)

class_labels=['Angry','Happy','Neutral','Sad','Surprise']


# image_without_alpha = main_body[:,:,:3]                             # 전체
# bg = np.full(image_without_alpha.shape,(0,215,0), dtype=np.uint8)   # 크로마키 배경이미지

def make_body(main_body,shoe=None,pants=None,cloth=None,another1=None,another2=None) :
    bg = rd.make_bg(main_body)
    image_stack = [main_body]

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

    for image in image_stack :
        bg = Image.alpha_composite(bg, image)

    return bg


def make_head(main_head,hair = None,another1 = None,another2=None) :
    image_stack=[]
    if hair :
        image_stack.append(hair)
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
def make_face(eyes123=[],mouth123=[]) :
    face=[]
    emotion = []
    mouths = []
    i = 1
    for eye in range(len(eyes)) :
        for mou in range(len(mouth)) :
            if eye//3 != mou//4 :
                mou = (eye//3)*4+mou
            # 여기에 얼굴과 입 합성 함수
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


    # main_body = Image.alpha_composite(main_body, main_head)
    # main_body = Image.alpha_composite(bg, main_body)
    body = make_body(main_body, cloth=cloth)
    head = make_head(main_head, hair=hair)
    # main_body.paste(main_head,(0,0))
    # body.show()
    # head.show()
    frame1 = Image.alpha_composite(body,head)

    shift_head = ImageChops.offset(head,0,30)
    frame2 = Image.alpha_composite(body,shift_head)

    frame1.show()
    frame2.show()


    # elapsed = (time.perf_counter() - start_time) * 1000
    # print ( "수행시간 = %0.2f ms" % elapsed)
    # cv2.waitKey(0) & 0xFF == 27

if __name__ == "__main__":
    main()
