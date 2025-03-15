import pickle

import cv2
import numpy as np
mthickness = -1
def draw_haar_feature(image, feature, polarity, threshold):
    """绘制Haar特征"""
    feat_type, x, y, w, h = feature
    x, y, w, h = x*10, y*10, w*10, h*10
    if polarity == 1:
        black, white = (0, 0, 0), (255, 255, 255)
    else:
        black, white = (255, 255, 255), (0, 0, 0)



    # Draw the feature based on its type
    if feat_type == 'two_horizontal':
        w_half = w // 2
        # Draw two rectangles horizontally
        cv2.rectangle(image, (x, y), (x + w_half, y + h), white, mthickness)  # First rectangle
        cv2.rectangle(image, (x + w_half, y), (x + w, y + h), black, mthickness)  # Second rectangle

    elif feat_type == 'two_vertical':
        h_half = h // 2
        # Draw two rectangles vertically
        cv2.rectangle(image, (x, y), (x + w, y + h_half), black, mthickness)  # First rectangle
        cv2.rectangle(image, (x, y + h_half), (x + w, y + h), white, mthickness)  # Second rectangle

    elif feat_type == 'three_horizontal':
        w_third = w // 3
        # Draw three rectangles horizontally
        cv2.rectangle(image, (x, y), (x + w_third, y + h), white, mthickness)  # First rectangle
        cv2.rectangle(image, (x + w_third, y), (x + 2 * w_third, y + h), black, mthickness)  # Second rectangle
        cv2.rectangle(image, (x + 2 * w_third, y), (x + w, y + h), white, mthickness)  # Third rectangle

    elif feat_type == 'three_vertical':
        h_third = h // 3
        # Draw three rectangles vertically
        cv2.rectangle(image, (x, y), (x + w, y + h_third), white, mthickness)  # First rectangle
        cv2.rectangle(image, (x, y + h_third), (x + w, y + 2 * h_third), black, mthickness)  # Second rectangle
        cv2.rectangle(image, (x, y + 2 * h_third), (x + w, y + h), white, mthickness)  # Third rectangle

    elif feat_type == 'four_rect':
        w_half = w // 2
        h_half = h // 2
        # Draw four rectangles
        cv2.rectangle(image, (x, y), (x + w_half, y + h_half), white, mthickness)  # First rectangle
        cv2.rectangle(image, (x + w_half, y), (x + w, y + h_half), black, mthickness)  # Second rectangle
        cv2.rectangle(image, (x, y + h_half), (x + w_half, y + h), black, mthickness)  # Third rectangle
        cv2.rectangle(image, (x + w_half, y + h_half), (x + w, y + h), white, mthickness)  # Fourth rectangle

    else:
        raise ValueError("未知特征类型")

    return image


model_path = R'output/stage_2_classifier.pkl'

import sys

# Example Usage
if __name__ == "__main__":

    args = sys.argv
    if len(args) >= 2:
        model_path = args[1]

    with open(model_path, 'rb+') as f:
        features = pickle.load(f)

    # Load a sample image (or create a blank image)
    img = np.ones((240, 240, 3), dtype=np.uint8) * 150  # White canvas
    i = 0
    # Define a Haar feature
    for feat in features:

        # Draw the Haar feature
        img_with_feature = draw_haar_feature(img.copy(), feat['feature'], feat['polarity'], feat['threshold'])

        # Show the image
        cv2.namedWindow('Haar Feature', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Haar Feature', 480, 480)
        cv2.imshow('Haar Feature', img_with_feature)
        cv2.imshow('Haar Feature', img_with_feature)
        cv2.imwrite(f'feat{i}.jpg', img_with_feature)
        i+=1
        print(feat)
        if cv2.waitKey(0) == ord('q'):
            exit()
        cv2.destroyAllWindows()

