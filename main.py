import cv2, dlib, sys
import numpy as np

scaler = 1

# initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load video
cap = cv2.VideoCapture('samples/man.mp4')
overlay_img = cv2.imread('samples/kakao.png', cv2.IMREAD_UNCHANGED)


# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
  bg_img = background_img.copy()
  # convert 3 channels to 4 channels
  if bg_img.shape[2] == 3:
    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

  if overlay_size is not None:
    img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

  b, g, r, a = cv2.split(img_to_overlay_t)

  mask = cv2.medianBlur(a, 5)

  h, w, _ = img_to_overlay_t.shape
  roi = bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

  img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
  img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

  bg_img[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = cv2.add(img1_bg, img2_fg)

  # convert 4 channels to 4 channels
  bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

  return bg_img


while True:
  ret, img = cap.read()
  if not ret:
    break

# 영상의 회전각에 따라 얼굴 인식이 불가한 경우가 있으므로
# 영상이 -90 회전이 되어있는 경우 이미지를 90도 회전
  #img = cv2.transpose(img)
  #img = cv2.flip(img, 1)

  img = cv2.resize(img, (int(img.shape[1] * scaler), int(img.shape[0] * scaler)))
  ori = img.copy()
  faces = detector(img)

  if len(faces) > 0:
    for i in list(faces):
      img = cv2.rectangle(img,
                          pt1=(i.left(), i.top()),
                          pt2=(i.right(), i.bottom()),
                          color=(255, 255, 255),
                          thickness=2,
                          lineType=cv2.LINE_AA)

      # Head Shape
      dlip_shape = predictor(img, i)
      shape_2d = np.array([[p.x, p.y] for p in dlip_shape.parts()])

      # Head Size
      top_left = np.min(shape_2d, axis=0)
      bottom_right = np.max(shape_2d, axis=0)
      face_size = max(bottom_right - top_left)

      shape_index = 0

      for s in shape_2d:
        cv2.circle(img,
                   center=tuple(s),
                   radius=1,
                   color=(255, 255, 255),
                   thickness=1,
                   lineType=cv2.LINE_AA)

        shape_index += 1
        cv2.putText(img,
                    '%d' %shape_index,
                    tuple(s),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)

      center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)
      result = overlay_transparent(ori,
                                   overlay_img,
                                   center_x, center_y,
                                   overlay_size=(face_size, face_size))

      cv2.imshow('imgOverlay', result)

  cv2.imshow('img', img)
  cv2.waitKey(20)













