import tensorflow.lite as tf
import cv2
import numpy as np
import time


def load_model(model_location):
    interpreter = tf.Interpreter(model_path=model_location)
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
    predictions = invoke_interpreter(interpreter, tensor)
    predictions = np.argmax(predictions)
    text = predictions

    return text


def stream_and_predict(model, video):
    while True:
        ret, frame = video.read()
        start_time = time.time()
        prediction = predict(model, frame)
        duration = time.time() - start_time

        cv2.putText(frame, str(prediction), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, str(1/duration) + " fps", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q') & 0xFF:
            break


def main():
    interpreter = load_model("../layered_model.tflite")
    vid = cv2.VideoCapture(0)
    stream_and_predict(interpreter, vid)
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
