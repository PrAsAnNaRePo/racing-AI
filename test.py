import time
import cv2
import keras.models
import numpy as np
from directkeys import PressKey, W, ReleaseKey, A, D
from get_keys import key_check
from grab_screen import grab_screen

model = keras.models.load_model('roi-color-3var-Xception-0.001-35e')

vertices = np.array([[0,180],[10,90], [100,70], [200,70], [230,90], [240,180]], np.int32)

t_time = 0.09

def straight():
##    if random.randrange(4) == 2:
##        ReleaseKey(W)
##    else:
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(A)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    #ReleaseKey(W)
    #ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(D)


def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_img(original_image):
    processed_img = cv2.resize(original_image, (240, 180))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[0, 420], [10, 260], [150, 200], [350, 200], [500, 260], [460, 420]], np.int32)
    processed_img = roi(processed_img, [vertices])
    return processed_img


def main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    while True:

        if not paused:
            screen = grab_screen(region=(0, 0, 1920, 1080))
            resized_img = cv2.resize(screen, (240, 180))
            processed_img = roi(resized_img, [vertices])
            img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            images = np.array([img])
            prediction = model.predict(images)[0]
            # print(pred)
            turn_thresh = .75
            fwd_thresh = 0.70

            if prediction[1] > fwd_thresh:
                straight()
            elif prediction[0] > turn_thresh:
                left()
            elif prediction[2] > turn_thresh:
                right()
            else:
                straight()
            # forward_precrent = pred[1] * 100
            # left_precrent = pred[0] * 100
            # right_precrent = pred[2] * 100
            # print(left_precrent, forward_precrent, right_precrent)
            #
            # if forward_precrent >= 50:
            #     if forward_precrent >= 75:
            #         ReleaseKey(A)
            #         ReleaseKey(D)
            #         PressKey(W)
            #         # print('straight')
            #     if forward_precrent <= 74 and left_precrent > right_precrent:
            #         ReleaseKey(D)
            #         ReleaseKey(A)
            #         PressKey(W)
            #         PressKey(A)
            #         time.sleep(0.25)
            #         ReleaseKey(A)
            #         # print('straights and bit left')
            #     if forward_precrent <= 74 and  right_precrent > left_precrent:
            #         ReleaseKey(A)
            #         ReleaseKey(D)
            #         PressKey(W)
            #         PressKey(D)
            #         time.sleep(0.25)
            #         ReleaseKey(D)
            #         # print('straights and bit right')
            #
            # if left_precrent >= 52:
            #     if left_precrent >= 84:
            #         ReleaseKey(W)
            #         ReleaseKey(D)
            #         PressKey(A)
            #         print('left')
            #     elif left_precrent <= 83 and forward_precrent > right_precrent:
            #         ReleaseKey(W)
            #         ReleaseKey(D)
            #         PressKey(A)
            #         PressKey(W)
            #         time.sleep(0.25)
            #         ReleaseKey(W)
            #         print('left and straight')
            # if right_precrent >= 52:
            #     if right_precrent >= 84:
            #         ReleaseKey(W)
            #         ReleaseKey(A)
            #         PressKey(D)
            #         print('right')
            #     if right_precrent < 83 and forward_precrent > left_precrent:
            #         ReleaseKey(W)
            #         ReleaseKey(A)
            #         PressKey(D)
            #         PressKey(W)
            #         time.sleep(0.25)
            #         ReleaseKey(W)
            #         print('rights and straight')

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpause!')
                time.sleep(1)
            else:
                print('Pausing!')
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                paused = True
                time.sleep(1)


main()
