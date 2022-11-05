import cv2

def get_img (location) :
    resize = 0.3
    img = cv2.imread(location, cv2.IMREAD_UNCHANGED) # png 에서 알파값을 받아오는 함수...
    img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
    return img

'''
def resize_pic(img) :
    resize = 1
    # img = cv2.resize(img, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA) #축소 한정
    # img = cv2.resize(img, None, resize, resize, interpolation=cv2.INTER_NEAREST)    # 확대시 속도우선
    # img = cv2.resize(img, None, resize, resize, interpolation=cv2.INTER_LINEAR)     # 확대시 속도중 화질중
    # img = cv2.resize(img, None, resize, resize, interpolation=cv2.INTER_CUBIC)      # 확대시 속도하 화질상

    return img
'''