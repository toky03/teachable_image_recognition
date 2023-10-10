import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from gpiozero import LED
import time

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.adapters import classify
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference



def load_model(model_location):
    interpreter = make_interpreter(model_location)
    interpreter.allocate_tensors()  
    return interpreter


def sigmoid(x):
    return 1/(1+np.exp(-x))


def cv2_to_tensor(cv2_im, inference_size):
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    return cv2.resize(cv2_im_rgb, inference_size)


def predict(interpreter, frame, labels: dict) -> str:
    common.set_input(interpreter, frame)
    interpreter.invoke()
    detected_classes = classify.get_classes(interpreter, 2, 0.5)
    print(detected_classes)
    if(len(detected_classes) < 1): 
        return ''
    return str(labels.get(detected_classes[0], detected_classes[0]))

def detect_boxes(interpreter, frame, inference_size):
    cv2_im = frame
    tensor = cv2_to_tensor(cv2_im, inference_size)
    
    run_inference(interpreter, tensor.tobytes())
    # index 16 is cat
    return list(filter(lambda x: x.id == 16, detect.get_objects(interpreter, 0.5)))[:2]


def classify_frame(model, frame, objs, labels, inference_size):
    height, width, channels = frame.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    detections = []
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        cropped_frame = frame[y0:y1, x0:x1]
        
        if cropped_frame.size == 0:
            continue
        cv2.imshow('cropped', cropped_frame)
        normalized_size = cv2.resize(cropped_frame, inference_size, interpolation=cv2.INTER_AREA)
        prediction = predict(model, normalized_size, labels)
        detections.append(prediction)
        
    return detections



def stream_and_predict(classification_model,object_detection_model, video, led):
    object_detection_size = common.input_size(object_detection_model)
    classification_size = common.input_size(classification_model)
    category_labels = read_label_file('../object_detection_models/coco_labels.txt')
    classification_labels = read_label_file('../classification_model/labels.txt')
    print('Start Prediction')
    checker = FrameChecker(consecutive_frames=4, tolerance_frames=1)
    
    while True:
        ret, frame = video.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        start_time = time.time()
        boxes = detect_boxes(object_detection_model, frame, object_detection_size)
        detections = classify_frame(classification_model, frame, boxes, classification_labels, classification_size )    
        condition_met =  'caliou' in detections or 'nora' in detections
        duration = time.time() - start_time
        print(str(1/duration) + " fps")

        if checker.check_condition(condition_met):
            led.on()
        else:
            led.off()

        if cv2.waitKey(1) == ord('q') & 0xFF:
            break


def main():
    led = LED(17)
    led.off()
    classification_interpreter = load_model("../classification_model/layered_model_edgetpu.tflite")
    object_detection_interpreter = load_model("../object_detection_models/object_detection_edgetpu.tflite")
    vid = cv2.VideoCapture(-1)
    stream_and_predict(classification_interpreter, object_detection_interpreter, vid, led)
    vid.release()
    cv2.destroyAllWindows()



class FrameChecker:
    def __init__(self, consecutive_frames=10, tolerance_frames=2):
        self.consecutive_frames = consecutive_frames
        self.tolerance_frames = tolerance_frames
        self.frames = []

    def check_condition(self, condition):
        self.frames.append(condition)

        if len(self.frames) > self.consecutive_frames:
            self.frames.pop(0)

        true_frames = sum(self.frames)

        if true_frames >= self.consecutive_frames - self.tolerance_frames:
            return True  # Condition met for 10 consecutive frames with a tolerance of 2
        else:
            return False


if __name__ == '__main__':
    main()
