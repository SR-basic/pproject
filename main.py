import cv2
import mediapipe as mp
import utils
import numpy as np

test_mode = True

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE=[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 랜드마크를 찾아서 표시하는 함수, draw=true면 얼굴에 점을 찍는다.
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # 랜드마크의 [x,y]값에 캡쳐한 화면의 가로와 세로의 값을 곱해야 사용 가능
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # 각 랜드마크의 x,y좌표를 mesh_coord에 넣어 return시킨다.
    return mesh_coord

def test_draw_eyeline(img,mesh_coords,test_mode):
    if test_mode :
        cv2.polylines(img, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                      cv2.LINE_AA)
        cv2.polylines(img, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN, 1,
                      cv2.LINE_AA)
    return 0

# 아래 주석처리는 opencv의 기본 랜드마크 읽기로 테스트할때 주석 해제
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0) #캡쳐되는 이미지 변수 : cap


if __name__ == "__main__" :         # main 함수
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

            frame = cv2.resize(frame, None, fx=1.5, fy=1.5,
                                   interpolation=cv2.INTER_CUBIC)  # 원본의 가로 세로 fx,fy 배율, 리사이징 값
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(frame, results, False) # True면 감지된 모든 랜드마크를 바로바로 보여줍니다.
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

                # 보기 편하게 이미지를 좌우 반전합니다.
                cv2.imshow('MediaPipe Face Mesh(Puleugo)', cv2.flip(frame, 1))
                if cv2.waitKey(5) & 0xFF == 27:  # esc가 눌렸을 경우 종료
                    break



    cap.release()               # 웹캠의 정상적인 종료를 위해 반드시 첨부
    cv2.destroyAllWindows()
