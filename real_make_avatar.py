import cv2
import numpy as np
import img_reading as rd
import time


main_body = rd.get_full_img('./img/body/main_body.png')
main_head = rd.get_full_img('./img/body/main_head.png')

image_without_alpha = main_body[:,:,:3]                             # 전체
bg = np.full(image_without_alpha.shape,(0,215,0), dtype=np.uint8)   # 크로마키 배경이미지

def merge_img() :
    return 0

def main():

    # start_time = time.perf_counter()
    cv2.imshow("bg", bg)
    cv2.imshow("body", main_body)
    cv2.imshow("head", main_head)

    # elapsed = (time.perf_counter() - start_time) * 1000
    # print ( "수행시간 = %0.2f ms" % elapsed)
    cv2.waitKey(0) & 0xFF == 27

if __name__ == "__main__":
    main()
