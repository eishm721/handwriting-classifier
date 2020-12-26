from cv2 import cv2
import numpy as np 
import tensorflow as tf
import testModel
from settings import *

class DrawingGUI:
    """
    GUI for drawing B&W images on openCV canvas and making predictions
    on handwriting
    """
    def __init__(self):
        self.thickness = PIXEL_SIZE * 2
        self.img = np.zeros((BIG_WIDTH, BIG_WIDTH), np.uint8)
        self.drawing = False   # true if mouse is pressed
        self.pt1_x = self.pt1_y = None
        self.color = WHITE_COLOR

    def toggleEraser(self):
        """
        Swaps color of paintbrush between black and white
        """
        self.color = BLACK_COLOR if self.color == WHITE_COLOR else WHITE_COLOR

    def clearCanvas(self):
        """
        Clears canvas to all black
        """
        self.img = np.zeros((BIG_WIDTH, BIG_WIDTH), np.uint8)

    def lineDrawing(self, event, x, y, flags, param):
        """
        Callback function to handle mouse UI and drawing GUI
        """
        # clicking down on mouse
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.pt1_x, self.pt1_y = x, y
        # move mouse
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.line(self.img,(self.pt1_x, self.pt1_y),(x, y), color=self.color, thickness=self.thickness)
                self.pt1_x, self.pt1_y = x, y   
        # release mouse click  
        elif event==cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.line(self.img, (self.pt1_x, self.pt1_y), (x, y), color=self.color, thickness=self.thickness)        

    def getResizedImg(self):
        """
        Resizes image from large canvas in high resolution to 28 x 28 picture
        for neural network.
        Searches large canvas and if a pixel is on, turns corresponding pixel in small canvas on
        """
        genPixels = lambda sz: tuple([(i,j) for i in range(sz) for j in range(sz)])
        new = np.zeros((COMPRESSED_WIDTH, COMPRESSED_WIDTH))
        for pixel in genPixels(COMPRESSED_WIDTH):
            numOn = 0
            for subPixel in genPixels(PIXEL_SIZE):
                row = PIXEL_SIZE * pixel[0] + subPixel[0]
                col = PIXEL_SIZE * pixel[1] + subPixel[1]
                if self.img[row][col]:
                    numOn += 1
            # set new pixel brightness proportional to number of white pixels in original image
            new[pixel] = int(WHITE_COLOR * (numOn / (PIXEL_SIZE ** 2)))

        # show converted image - for testing
        # cv2.namedWindow('a')
        # cv2.imshow('a', new)
        return new


def showInstructions():
    """
    Prints instructions on how to use app
    """
    print("\nPress ESC or close window to stop")
    print("Press SPACE to make prediction with current canvas")
    print("Press Q to clear canvas")
    print("Press E to toggle eraser")


def main():
    """
    Main function to handle OpenCV windows and neural engine
    """
    # link openCV canvas with drawing GUI callback
    canvas = DrawingGUI()
    cv2.namedWindow('imageDetector')
    cv2.setMouseCallback('imageDetector', canvas.lineDrawing)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # disable TF optimization warnings
    c = testModel.HandwritingClassifier()
    showInstructions()

    # runs while window is open
    while cv2.getWindowProperty('imageDetector', cv2.WND_PROP_VISIBLE) > 0:
        cv2.imshow('imageDetector', canvas.img)
        key = cv2.waitKey(33)
        if key == ESC_KEY:
            break
        elif key == Q_KEY:
            canvas.clearCanvas()
        elif key == E_KEY:
            canvas.toggleEraser()
        elif key == SPACE_KEY:
            # run neural network on current canvas 
            predictions = c.predict(canvas.getResizedImg())[:2]
            print()
            [print('{:15} Confidence: {:.5f}'.format(letter, round(prob, 5))) for letter, prob in predictions]
            print()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()