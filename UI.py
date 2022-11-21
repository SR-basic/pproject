import cv2 as cv
import customtkinter
import tkinter
from tkinter import *
import main



# 기본설정
customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"

def show_frame(on_photo=None,off_photo=None):
    cap = cv.VideoCapture(0)
    if cap:
        ret, frame = cap.read()
        print(ret)
        if ret:
            image = on_photo
        else :
            image= off_photo
    else :
        image= off_photo

    return image

    # vid_lbl.after(100, show_frame)


def active_mode():
    cam = False
    main.main(cam)


def test_mode():
    cam = True
    main.main(cam)



def start_UI() :

        # mainwindow 생성
        mainWindow = customtkinter.CTk()
        # mainWindow2 = Toplevel()
        mainWindow.geometry("800x540")
        mainWindow.title("Deep Team")
        mainWindow.minsize(790,530)
        mainWindow.maxsize(800,540)

        # mainframe 생성
        bg = tkinter.PhotoImage(file='bg.png')
        canvas=Canvas(mainWindow,width=800,height=540)
        canvas.pack(fill="both",expand=True)
        canvas.create_image(0,0,image=bg,anchor="nw")
        mainFrame=canvas

        # mainFrame = customtkinter.CTkLabel(master=mainWindow,image=bg)
        # # mainFrame.place(x=350, y=0)
        # mainFrame.pack(pady=10, padx=20, fill="both", expand=True)

        # 왼쪽에 카메라가 있는지 확인 함수
        # vid_lbl = customtkinter.CTkLabel(mainFrame, text="")
        # vid_lbl.place(x=0, y=0)

        on_photo = tkinter.PhotoImage(file='test.png')
        off_photo = tkinter.PhotoImage(file='test2.png')
        image = show_frame(on_photo,off_photo)
        img_label = customtkinter.CTkLabel(mainFrame, image = image)
        img_label.grid(row=0,column=0,padx=30,pady=20,sticky='nw')

        shadow_frame = customtkinter.CTkFrame(master=mainFrame,width=0,height=200)
        shadow_frame.grid(row=1,column=0,columnspan=2,sticky="w")

        # Buttons
        TurnCameraOn = customtkinter.CTkButton(mainFrame, command=test_mode, text="Test mode", height=120, width=300)
        # TurnCameraOn.pack(padx=10,pady=30)
        TurnCameraOn.grid(row=2,column=0,padx=30,pady=50,sticky='ew')
        TurnCameraOff = customtkinter.CTkButton(mainFrame, command=active_mode, text="Active mode", height=120, width=300)
        # TurnCameraOff.pack(pady=12, padx=10)
        TurnCameraOff.grid(row=2,column=1,padx=30,pady=50,sticky='nw')

        button_3 = customtkinter.CTkButton(mainFrame, text="avatar mode(미구현 ㅎ..)", height=60, width=160)
        # button_3.pack(side="right",pady=12, padx=10)
        button_3.grid(row=0,column=1,padx=30,pady=20,sticky='ne')

        mainWindow.mainloop()



if __name__ == "__main__":
    start_UI()
