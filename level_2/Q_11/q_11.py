import cv2
import numpy as np
import pytesseract
import imutils

def preprocess_image(image):
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    return image, gray, edged

def perform_ocr(cropped_image):
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(cropped_image, config=config)
    return text.strip()

def main():
    original_img = cv2.imread("licence-plate.jpg")
    
    image, gray, edged = preprocess_image(original_img)

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    location = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        if len(approx) == 4:
            location = approx
            break

    if location is not None:
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [location], 0, 255, -1)
        
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        cropped = gray[topx:bottomx+1, topy:bottomy+1]

        plate_text = perform_ocr(cropped)
        print("Detected Plate:", plate_text)

        cv2.drawContours(image, [location], -1, (0, 255, 0), 3)
        cv2.imshow("Detection", image)
        # cv2.imshow("Cropped Plate", cropped)
    else:
        print("Could not detect a 4-sided polygon.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()