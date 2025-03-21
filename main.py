import cv2
import mediapipe as mp
import utils, math
import avatar
import real_make_avatar as mk
import numpy as np
import UI
import tkinter_opencv
from keras.models import load_model
from time import sleep
from keras_preprocessing.image import img_to_array
from keras_preprocessing import image

'''
진행할 내용
1. 애니메이션 효과
 표정이 변할떄 눈은 ok지만 
 감정이 변할떄 그 사이의 0프레임의 어색함, 입의 움직임도 마찬가지
 사람이 숨쉴때 가만있더라도 미세히 움직이듯 캐릭터가 8자형태로 미세하게 움직이는 효과를 넣어볼까 함
 
2. tkinter와 opencv의 연동
 많은 상호작용을 하고싶어요 
 예를들어 스트리머의 보호를 위해 카메라가 검게되어있는 곳을 원한다면 경고창을 한번 띄워준 후 실제 캠 영상을 보여준다던가
 캐릭터의 상반신과 전신을 오가게 만든다던가
 그러기 위해서 tkinter의 연동의 필요함
 
---------

3. 아바타 모드의 구현
 현재 프로그램은 폴더안에 들어있는 png파일을 바꿔야지 아바타 레이어를 변경할 수 있음
 이런 레이어들을 모아놓는 폴더하나와, 사용자가 고른 레이어들을 실행하기위해 모아놓은 폴더로 분리한 후,
 ui우측 상단의 미구현버튼인 아바타모드를 구현, 원하는 아바타 레이어를 고를 시, 실시간으로 그 레이어를 입힌 모습을 보여주고,
 결정을 하면 그걸 실제 프로그램에서 사용 가능하도록 설계!!
 
4. 파일을 어떻게 다룰지 결정
 eye 폴더에 eye1,eye2,eye3이렇게 모아두고, 머리카락폴더에 hair1,hai2....이런식으로 모아두는게 코딩할때 편할까?
 사용자 입장에선 '커스텀 아바타'파일 하나를 올려놓으면 자동으로 압축해제해서 eye프레임은 eye폴더에, shoes프레임은 shoes폴더에 들어가게
 하면 좋을 듯한데... 이런 구조화는 어떡할지는 나중에 생각하고, 일단 아바타모드 구현에 들어가면 그떄 생각
 
 -------

코딩 외적 문제

UI의 이미지 파일 입히기(버튼 포함)

완성된 캐릭터 풀채색(힘줘서!) 하나 만들기

옷이 지금 테스트용이라 (있어보이려고) 상의와 하의가 한레이어로 되어있는데,
분리시키고, 상의에 가려서 안보이는 바지부분도 그려보기
글구 캐릭터 빌려주신분 꼬리있던데 꼬리를 안그렸어요 그거 추가좀(그럼 아바타 레이어 순위를 하나더 만들어야 하나?)


'''

cam = True          # 카메라 프로그램이 켜집니다. 즉 실사용시엔 False가 될 예정 (UI 실전모드)
test_mode = True   # 테스트 모드(기본값)
avatar_mode = True  # 판떼기 프로그램이 켜집니다 (UI 모든모드)
protect_mode = True    # 기본적으로 True가 되도록 짤 예정, 카메라를 가려서 흰색만 보이지만, 테스트모드의 True로 눈,입은 보입니다.

# 변수들
FRAME_COUNTER =0 # 카메라가 이미지를 받아오는 매 프레임을 수치화, 과부화 비중치가 높은 감정인식함수 지연호출
CEF_COUNTER =0   # 눈의 깜빡임에 관련된 변수, 눈을 감음 상태가 1프레임 감지될 때마다 1씩 추가된다.
TOTAL_BLINKS =0  # 눈을 몇번 깜빡였는 지 알 수 있는 변수, CEF_COUNTER의 값이 CLOSED_EYE_FRAME보다 값이 클시 1이 추가된다.

CLOSED_EYES_FRAME =3                # 눈을 깜빡였는지 판정하는 함수, n프레임만큼 감았다고 판정할시 시스템적으로 감았다 라고 판정
FONTS =cv2.FONT_HERSHEY_COMPLEX     # 카메라실행시 옆에보이는 글자들 폰트

blink_judge = 4.5        # 눈을 감은 판정을 결정하는 종횡비 수 (기본값 5.5)
blink_animation = 0      # 눈을 감은 판정이 떴을때 1, 아닐때 0, 2는 1과  0사이의 애니메이션효과
mouth_animation = 0      # 입의 움직임 0,1,2,3 숫자가 클수록 입의사이즈가 큼

detected_emotion = 5    # 감정인식 값, re_class_labels 참조

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH = [78, 191, 80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]

# 아래 세 변수는 얼굴인식을 위한 변수
# face_classifier=cv2.CascadeClassifier('./face_emotion/haarcascade_frontalface_default.xml')
# classifier = load_model('./face_emotion/EmotionDetectionModel.h5')

face_classifier=cv2.CascadeClassifier('./made_face_emotion/haarcascade_frontalface_default.xml')
classifier = load_model('./made_face_emotion/model_optimal.h5')
classifier.load_weights('./made_face_emotion/model_weights.h5')


class_labels=['Angry','Happy','Neutral','Sad','Surprise']
re_class_labels = ['Neutral','Happy','Sad','Angry','Surprise','None']



# 랜드마크를 찾아서 표시하는 함수, draw=true면 얼굴에 점을 찍는다.
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # 랜드마크의 [x,y]값에 캡쳐한 화면의 가로와 세로의 값을 곱해야 사용 가능
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # 각 랜드마크의 x,y좌표를 mesh_coord에 넣어 return시킨다.
    return mesh_coord

# 두 랜드마크 사이의 거리를 나타내는 함수
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio 눈의 깜박임을 측정하는 눈의 가로/세로 비율
def blinkRatio(img, landmarks, right_indices, left_indices):
    # 오른쪽 눈
    # 가로 라인
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # 세로 라인
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # 아래 주석을 해제하면 오른쪽눈의 가로라인과 세로라인을 확인 가능
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # 왼쪽 눈
    # 가로 라인
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # 세로 라인
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    # 눈이 완벽히 감겨서 아래 비율을 구할때 divided by zero 오류를 막기위해 해당 코드 추가
    if rvDistance == 0 :
        rvDistance = 0.001

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    if lvDistance == 0 :
        lvDistance = 0.001

    reRatio = rhDistance/rvDistance     #오른쪽 눈 비율
    leRatio = lhDistance/lvDistance     #왼쪽 눈 비율

    ratio = (reRatio+leRatio)/2         #평균값
    return ratio


# 눈의 감았다 판정이 종횡비추정으로 구하는것이 반응이 좋아 입도 같은방식으로 진행
# 종횡비로 구하는 이유는 카메라와 사람과의 거리(깊이)에 관계없이 판정을 낼 수 있는 수단이기 때문
def mouthRatio(img, landmarks, mouth_indices,test = False) :
    # 오른쪽 눈
    # 가로 라인
    mh_right = landmarks[mouth_indices[0]]
    mh_left = landmarks[mouth_indices[10]]
    # 세로 라인
    mv_top = landmarks[mouth_indices[5]]
    mv_bottom = landmarks[mouth_indices[15]]

    if test :
        cv2.line(img, mh_right, mh_left, utils.GREEN, 2)
        cv2.line(img, mv_top, mv_bottom, utils.WHITE, 2)

    mhDistance = euclaideanDistance(mh_right, mh_left)
    mvDistance = euclaideanDistance(mv_top, mv_bottom)

    # 입이 완벽히 감겨서 아래 비율을 구할때 divided by zero 오류를 막기위해 해당 코드 추가
    if mvDistance == 0 :
        mvDistance = 0.001

    ratio = mhDistance/mvDistance     #입의 비율

    return ratio

# 입 종횡비에 따라 어떤 입모양을 출력할 지 결정하는 함수 , 아바타모드가 켜져있다면 avatar.py에서 구현(예정)
# 레이어 합성을통한 캐릭터 생성 이후 애니메이션 구현예정
def mouth_judge (frame,mouth_ratio) :
    frame_height, frame_width = frame.shape[:2]
    mouth = 0       # mouth 값이 작을수록 입 모양이 작음
    if mouth_ratio <= 2 :
        utils.colorBackgroundText(frame, f'm.pic3 bgmouth', FONTS, 1.0, (int(frame_height / 2), 50), 2, utils.YELLOW, pad_x=6, pad_y=6, )
        mouth = 3
    elif mouth_ratio <= 8 :
        utils.colorBackgroundText(frame, f'm.pic2 mlmouth', FONTS, 1.0, (int(frame_height / 2), 50), 2, utils.YELLOW,
                                  pad_x=6, pad_y=6, )
        mouth = 2
    elif mouth_ratio <= 10 :
        utils.colorBackgroundText(frame, f'm.pic1 msmouth', FONTS, 1.0, (int(frame_height / 2), 50), 2, utils.YELLOW,
                                  pad_x=6, pad_y=6, )
        mouth = 1
    else :
        utils.colorBackgroundText(frame, f'm.pic0 nomouth', FONTS, 1.0, (int(frame_height / 2), 50), 2, utils.YELLOW,
                                  pad_x=6, pad_y=6, )
        mouth = 0

    return mouth
# test_mode일때만 작동, 출력중인 카메라에 눈을 선으로 그려줍니다.
def test_draw_eyeline(img,mesh_coords,test_mode) :
    if test_mode :
        cv2.polylines(img, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                      cv2.LINE_AA)
        cv2.polylines(img, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                      cv2.LINE_AA)
    return 0

def test_draw_mouth(img,mesh_coords,test_mode) :
    if test_mode :
        cv2.polylines(img, [np.array([mesh_coords[p] for p in MOUTH], dtype=np.int32)], True, utils.GREEN, 1,
                      cv2.LINE_AA)

    return 0

# 감정인식 함수, 카메라값을 받아들여 h5파일을 이용해 감정을 판단하고, 그 문자열을 반환
def emotion_detection(img,before_emotion) :
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)


    if len(faces) == 0 :
        emotion = before_emotion
    else :
        face = faces[0]
        (x,y,w,h) = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)        # 얼굴감지를 한 부분을 네모로 그리기
        utils.colorBackgroundText(img, f'emotion detected', FONTS, 0.5, (x, y), 2, textColor=(255,255,255))
        roi_gray = gray[y:y + h, x:x + w]                                   # 표정인식을 위한 gray_scale 변환
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA) # 학습모델과 비교하기 위해 같은사이즈로 변환

        if np.sum([roi_gray]) != 0:                 # 즉, 관심영역이 비어있는 상태가 아니면 인식 진행
            roi = roi_gray.astype('float') / 255.0  # 픽셀값 0~255을 계산하기위해 0~1의 값으로 만듬 + int > float
            roi = img_to_array(roi)                 # 계산을 위해 이미지를 배열로 변환
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]      # 딥러닝 자료를 가지고 판별
            if preds.argmax() == 0 :
                emotion = 3
            elif preds.argmax() == 1 :
                emotion = 1
            elif preds.argmax() == 2:
                emotion = 0
            elif preds.argmax() == 3 :
                emotion = 2
            elif preds.argmax() == 4 :
                emotion = 4
        else:                                       # 관심영역이 비어있는 상태면 인식 실패
            emotion = 5
    return emotion


# 아래 주석처리는 opencv의 기본 랜드마크 읽기로 테스트할때 주석 해제
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)



def main(cam = False) :        # main 함수
    cap = cv2.VideoCapture(0)  # 캡쳐되는 이미지 변수 : cap
    # cam = UI.start_UI()

    images = mk.main()

    global detected_emotion, FRAME_COUNTER, TOTAL_BLINKS,CEF_COUNTER, blink_animation
    # 이곳에 아바타를 먼저 선 생성하고 메인함수 구현 시작?
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()  # 웹캠을 탐지, 본격적인 실행문구 시작  # ret = 웹캠 탐지여부 true, frame
            if not cap.isOpened() :                 # 웹캠을 탐지하지 못하면 오류발생
                raise IOError("웹캠 찾지 못함.")
                break
            FRAME_COUNTER += 1
            frame = cv2.resize(frame, None, fx=1.5, fy=1.5,
                                   interpolation=cv2.INTER_CUBIC)  # 원본의 가로 세로 fx,fy 배율, 리사이징 값
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)

            if FRAME_COUNTER %10 == 0 :     # 감정 감지 함수입니다!! 과부하가 꽤 되는 함수라 0.3초마다 한번씩 호출
                detected_emotion = emotion_detection(frame,detected_emotion)
            if protect_mode :
                cv2.rectangle(frame,(0,0),(frame_width,frame_height),(0,0,0),cv2.FILLED)
            utils.colorBackgroundText(frame, f'emotion detected : {re_class_labels[detected_emotion]}', FONTS, 0.7,(30, 240), 2) #감정감지 함수에서 이어지는 함수. 원래대로라면 감정감지 직후 그 값을 바로 표시해야하지만,
                                                    #protect_mode가 켜져있다면 이 텍스트를 먹어버리기에 protect_mode를 선출력한뒤 텍스트 출력

            if results.multi_face_landmarks:                            # 여러 얼굴을 감지하도록 되어있지만 구현 구조상 1로 고정한다.
                for face_landmarks in results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(frame, results, False) # True면 감지된 모든 랜드마크를 바로바로 보여줍니다.
                    eye_ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                    utils.colorBackgroundText(frame, f'Eye Ratio : {round(eye_ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                              utils.YELLOW)

                    # blink_judge값을 변경하여 눈인식의 개인차 조정 가능
                    #
                    if eye_ratio > blink_judge:
                        CEF_COUNTER += 1
                        blink_animation = 1
                        # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                        utils.colorBackgroundText(frame, f'Blink!', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                    # else함수에 있는 내용은 깜빡인 횟수를 카운팅하는 내용, 눈의 애니메이션을 넣는 활동서 사용
                    else :
                        if CEF_COUNTER > CLOSED_EYES_FRAME:
                            TOTAL_BLINKS += 1
                            CEF_COUNTER = 0
                            blink_animation = 2

                    # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
                    utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)



                    mouth_ratio = mouthRatio(frame,mesh_coords, MOUTH, test=False)
                    utils.colorBackgroundText(frame, f'Mouth Ratio : {round(mouth_ratio, 2)}', FONTS, 0.7, (30, 200), 2,
                                              utils.PINK, utils.YELLOW)
                    mouth_animation = mouth_judge(frame,mouth_ratio)


                    if avatar_mode :
                        # 평상시 눈을 뜬 이미지, 눈을 감은 이미지 애니메이션 구현
                        avatar.show_avatar(blink_animation,mouth_animation,detected_emotion,images)
                        # if detected_emotion == 5:
                        #     detected_emotion == 0
                        #
                        # cv2.imshow('avatar', images[detected_emotion, blink_animation, mouth_animation])
                        #
                        # if blink_animation == 2:
                        #     cv2.waitKey(70)


                    blink_animation = 0


                    # 아래는 극 초기 mediapipe에서 제공하는 기본 툴로 간단하게 카메라가 돌아가고 얼마나 감지했다 는 내용을 안 내용
                    # 얼굴감지를 수동으로 구현하였으므로 현재는 사용되지 않음, 하지만 일단 남겨봄
                    # 만약 가상 캐릭터와 매칭을 시킨다면 아래내용은 제외

                    '''
                    mp_drawing.draw_landmarks(
                        image=image,        
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())        
                    '''
                    '''
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    '''

                    '''
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
                    '''

                    test_draw_eyeline(frame, mesh_coords, test_mode)
                    test_draw_mouth(frame, mesh_coords, test_mode)

                if cam :
                    cv2.imshow('testmode', frame)
                    # tkinter_opencv.UI_opencv(frame,name='testmode')
                    # cv2.waitKey(25) #자연스러운 애니메이션을 위해 부하를 넣어봤음...
                if cv2.waitKey(2) & 0xFF == 27:  # esc가 눌렸을 경우 종료
                    break



    cap.release()               # 웹캠의 정상적인 종료를 위해 반드시 첨부
    cv2.destroyAllWindows()


if __name__ == "__main__" :
    main()