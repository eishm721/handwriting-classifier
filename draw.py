from cv2 import cv2
from settings import *
import numpy as np 

class Drawer:
    def __init__(self):
        self.thickness = THICKNESS
        self.img = np.zeros((BIG_WIDTH, BIG_WIDTH), np.uint8)
        self.drawing = False   # true if mouse is pressed
        self.pt1_x = self.pt1_y = None

    def clearCanvas(self):
        self.img = np.zeros((BIG_WIDTH, BIG_WIDTH), np.uint8)

    # mouse callback function
    def lineDrawing(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1_x, self.pt1_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.img,(self.pt1_x, self.pt1_y),(x, y), color=WHITE, thickness=self.thickness)
                self.pt1_x, self.pt1_y = x, y     
        elif event==cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.img, (self.pt1_x, self.pt1_y), (x, y), color=WHITE, thickness=self.thickness)        


def resizeImg(img):
    """
    Resizes image from large canvas in high resolution to 28 x 28 picture
    for neural network.
    Searches large canvas and if a pixel is on, turns corresponding pixel in small canvas on
    """
    genPixels = lambda sz: tuple([(i,j) for i in range(sz) for j in range(sz)])
    new = np.zeros((PIXEL_SIZE, PIXEL_SIZE))
    for pixel in genPixels(PIXEL_SIZE):
        for subPixel in genPixels(THICKNESS):
            row = THICKNESS * pixel[0] + subPixel[0]
            col = THICKNESS * pixel[1] + subPixel[1]

            # if pixel in original canvas is on, turn on corresponding pixel in new canvas
            if img[row][col]:
                new[pixel] = WHITE
                break 
    return new
                        

if __name__ == '__main__':
    print("Press ESC or close window to stop")
    print("Press SPACE to make prediction with current canvas")
    print("Press Q to clear canvas")

    canvas = Drawer()
    cv2.namedWindow('imageDetector')
    cv2.setMouseCallback('imageDetector', canvas.lineDrawing)

    # runs while window is open
    while cv2.getWindowProperty('imageDetector', cv2.WND_PROP_VISIBLE) > 0:
        cv2.imshow('imageDetector', canvas.img)
        key = cv2.waitKey(33)
        if key == ESC_KEY:
            break
        elif key == Q_KEY:
            canvas.clearCanvas()
        elif key == SPACE_KEY:
            # run neural engine 
            converted = resizeImg(canvas.img)

    cv2.destroyAllWindows()