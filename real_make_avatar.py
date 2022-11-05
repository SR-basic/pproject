import cv2
import numpy as np
import img_reading as rd


main_body = rd.get_img('./img/body/main_body.png')
main_head = rd.get_img('./img/body/main_head.png')

def main():
    while True:
        cv2.imshow("body", main_body)
        cv2.imshow("head", main_head)

        if cv2.waitKey(2) & 0xFF == 27:  # esc가 눌렸을 경우 종료
            break

if __name__ == "__main__":
    main()
