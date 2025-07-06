import torch
import cv2
import time
from IPython import display

# load the trained YOLOv5 model 
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt')
model.conf = 0.85  #confidence threshold

#arabic class mapping
class_to_letter = {
    0: 'ع', 1: 'ى', 2: 'ا', 3: 'ب', 4: 'د', 5: 'ظ', 6: 'ض', 7: 'ف',
    8: 'ق', 9: 'غ', 10: 'ه', 11: 'ح', 12: 'ج', 13: 'ك', 14: 'خ',
    15: 'لا', 16: 'ل', 17: 'م', 18: 'ن', 19: 'ر', 20: 'ص', 21: 'س',
    22: 'ش', 23: 'ت', 24: 'ط', 25: 'ث', 26: 'ذ', 27: 'ة', 28: 'و',
    29: ' ', 30: 'ي', 31: 'ز'
}

# map model outputs to Arabic letters
for idx in model.names:
    model.names[idx] = class_to_letter[idx]

#Initialize webcam
text = ""
cap = cv2.VideoCapture(0)
cond = None

while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)

    #show detection frame
    cv2.imshow("Sign Language Translator", results.render()[0])

    #if prediction exists
    if results.pred[0].shape[0] and results.pred[0].shape[1]:
        current_letter = class_to_letter[int(results.pred[0][0][-1])]

        # Append character if different from last
        if current_letter != text[-1:]:

            if current_letter != 'لا':  # 'لا' is used as a delete key
                text += current_letter
                display.clear_output(wait=True)
                print(text, end="\r")

        # Handle deletion with 'لا'
        if current_letter == 'لا':
            if cond is None or time.time() - cond >= 0.5:
                text = text[:-1]
                cond = time.time()
                display.clear_output(wait=True)
                print(text, end="\r")

    # quit on 'q'
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
