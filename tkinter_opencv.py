import cv2 as cv
import customtkinter
from PIL import Image, ImageTk
from tkinter import *
# import main

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

def UI_opencv(frame,name=None) :
    frame_height, frame_width = frame.shape[:2]

    mainWindow = customtkinter.CTk()
    mainWindow.geometry("1000x640")
    mainWindow.title(name)
    mainWindow.resizable(False,False)

    mainWindow.label=Label(mainWindow)
    mainWindow.label.grid()

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    mainWindow.label.imgtk = imgtk
    mainWindow.label.configure(image=imgtk)
    # Repeat after an interval to capture continiously

    mainWindow.update()
    mainWindow.mainloop()

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    ret, frame = cap.read()  # 웹캠을 탐지, 본격적인 실행문구 시작  # ret = 웹캠 탐지여부 true, frame
    if not cap.isOpened():  # 웹캠을 탐지하지 못하면 오류발생
        raise IOError("웹캠 찾지 못함.")

    rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    UI_opencv(rgb_frame,'test')
