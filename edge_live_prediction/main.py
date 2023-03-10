import tflite_runtime.interpreter as tflite
import cv2
import numpy as np
from gpiozero import LED


def load_model(model_location):
    interpreter = tflite.Interpreter(model_path=model_location)
    interpreter.allocate_tensors()
    return interpreter


def sigmoid(x):
    return 1/(1+np.exp(-x))


def cv2_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 160))
    image_array = np.asarray(image, dtype='float32')
    return np.expand_dims(image_array, axis=0)


def invoke_interpreter(interpreter, frame):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], frame)

    interpreter.invoke()

    return interpreter.get_tensor(output_details[0]['index'])


def predict(interpreter, frame):
    tensor = cv2_to_tensor(frame)
    predictions = invoke_interpreter(interpreter, tensor).flatten()
    predictions = sigmoid(predictions)
    pred = np.where(predictions < 0.5, 1, 0)

    return pred


def stream_and_predict(model, video, led):
    print('Start Prediction')
    while True:
        ret, frame = video.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        prediction = predict(model, frame)
        if prediction[0] > 0:
            led.on()
        else:
            led.off()

        if cv2.waitKey(1) == ord('q') & 0xFF:
            break


def main():
    led = LED(17)
    led.off()
    interpreter = load_model("../layered_model.tflite")
    vid = cv2.VideoCapture(-1)
    stream_and_predict(interpreter, vid, led)
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
