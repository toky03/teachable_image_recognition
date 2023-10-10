import cv2
import numpy as np
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
    return str(detected_classes[0])

def detect_boxes(interpreter, frame, inference_size):
    cv2_im = frame
    tensor = cv2_to_tensor(cv2_im, inference_size)
    
    run_inference(interpreter, tensor.tobytes())
    return list(filter(lambda x: x.id in [87, 0], detect.get_objects(interpreter, 0.5)))[:2]


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
        detections.append(predict(model, normalized_size, labels))
        
    return detections


def stream_and_predict(classification_model,object_detection_model, video):
    object_detection_size = common.input_size(object_detection_model)
    classification_size = common.input_size(classification_model)
    category_labels = read_label_file('../object_detection_models/coco_labels.txt')
    classification_labels = read_label_file('../classification_model/labels.txt')
    while True:
        ret, frame = video.read()
        start_time = time.time()
        boxes = detect_boxes(object_detection_model, frame, object_detection_size)
        detections = classify_frame(classification_model, frame, boxes, classification_labels, classification_size )    
        duration = time.time() - start_time
        frame = append_objs_to_img(frame, object_detection_size, boxes, detections, category_labels )


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q') & 0xFF:
            break

def append_objs_to_img(cv2_im, inference_size, objs, detections, labels: dict):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    if len(detections) < len(objs):
        return cv2_im
    for idx, obj in enumerate(objs):
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = detections[idx]


        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im


def main():
    classification_interpreter = load_model("../classification_model/layered_model_edgetpu.tflite")
    object_detection_interpreter = load_model("../object_detection_models/object_detection_edgetpu.tflite")
    vid = cv2.VideoCapture(0)
    stream_and_predict(classification_interpreter, object_detection_interpreter, vid)
    vid.release()
    cv2.destroyAllWindows()


class DetectedObject:
    def __init__(self, boxes, score, class_id, width, height):
        self.start_point = int(boxes[1] * width), int(boxes[0] * height)
        self.end_point = int(boxes[3] * width), int(boxes[2] * height)
        self.score = score * 100
        self.class_id = class_id


if __name__ == '__main__':
    main()
