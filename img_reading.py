import cv2
import numpy as np
from PIL import Image


def convert_PIL_to_CV (img) :
    numpy_image = np.array(img)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image

def convert_CV_to_PIL (img) :
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image

def make_bg (img,color = (0,215,0)) :
    bg_img = Image.new("RGBA",img.size,color)
    return bg_img

def get_full_img (location,resize = 0.3) :
    img = cv2.imread(location, cv2.IMREAD_UNCHANGED) # png 에서 알파값을 받아오는 함수...
    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
    # if face :
        # cut_face(img)
    return img

def get_full_img_verPIL (location, face = True) :
    img = Image.open(location) # png 에서 알파값을 받아오는 함수...
    (width, height) = ((img.width*3)//10,(img.height*3)//10)
    resize_img = img.resize((width,height),Image.HAMMING)

    if face :
        resize_img = resize_img.crop((0, 0, 673, 639))
    # resize_img.show()
    return resize_img

'''
def resize_pic(img) :
    resize = 1
    # img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA) #축소 한정
    # img = cv2.resize(img, None, resize, resize, interpolation=cv2.INTER_NEAREST)    # 확대시 속도우선
    # img = cv2.resize(img, None, resize, resize, interpolation=cv2.INTER_LINEAR)     # 확대시 속도중 화질중
    # img = cv2.resize(img, None, resize, resize, interpolation=cv2.INTER_CUBIC)      # 확대시 속도하 화질상

    return img
'''

def main():

    main_head = get_full_img_verPIL('./img/body/main_head.png',False)
    main_body = get_full_img_verPIL('./img/body/main_body.png',False)

    bg = make_bg(main_body)
    main_body = Image.alpha_composite(main_body,main_head)
    main_body = Image.alpha_composite(bg,main_body)

    # main_body.paste(main_head,(0,0))
    main_body.show()


if __name__ == "__main__":
    main()


'''
아래는 pil test
    # start_time = time.perf_counter()
    main_body = get_full_img_verPIL('./img/body/main_head.png')
    # main_body.show()
    CV_main_body = convert_PIL_to_CV(main_body)

    cv2.imshow("body", CV_main_body)
    cv2.waitKey(0) & 0xFF == 27

    img0_0 = cv2.imread('./merge_image/0_0.png', cv2.IMREAD_COLOR)
    img0_0 = cv2.resize(img0_0, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    cv2.imshow("body", img0_0)
    cv2.waitKey(0) & 0xFF == 27
    img0_0 = convert_CV_to_PIL(img0_0)
    print(img0_0.size)

    cropped_img = main_body.crop((0,0,673,639))
    cropped_img = convert_PIL_to_CV(cropped_img)
    cv2.imshow("body", cropped_img)
    cv2.waitKey(0) & 0xFF == 27

    # elapsed = (time.perf_counter() - start_time) * 1000
    # print ( "수행시간 = %0.2f ms" % elapsed)
'''