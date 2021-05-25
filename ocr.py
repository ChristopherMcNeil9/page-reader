import cv2
import numpy as np
from PIL import Image
from pytesseract import pytesseract

# get video feed
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print('No webcam detected')
else:
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # calculates the position of floating text on image, 1% to the right of the left edge, 4% raised above bottom edge
    position = (int(width-width*.99), int(height-height*.04))
    counter = 180
    x = np.linspace(1, 180, 180)
    # print(x)
    # alpha = 2**(((1/counter)*(x-(1/counter)))**3)-1
    alpha = 1/(1+2.71828**((45-x)*.25))
    # print(values)
    while True:
        _, img = cam.read()
        # img = cv2.flip(img, 1)
        if counter > 0:
            overlay = np.zeros(img.shape, np.uint8)
            overlay = cv2.putText(overlay, 'press enter to capture or escape to quit program', position, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            # alpha = counter/180
            img = cv2.addWeighted(img, 1, overlay, alpha[counter-1], 0)
            # img = cv2.addWeighted(img, 1, overlay, alpha, 0)

            counter -= 1
        cv2.imshow('webcam', img)
        selection = cv2.waitKey(1)
        # print(selection)
        # escape key to exit, enter key to capture image
        if selection == 27:
            break
        if selection == 13:
            cv2.imwrite('output.png', img)
            input_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(Image.fromarray(input_image))
            print(text.replace('\n\n', '\n').replace('\x0c', '').rstrip())
            # print('image taken')
    cam.release()
    cv2.destroyAllWindows()

