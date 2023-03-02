import tensorflow as tf
import cv2
import numpy as np


def load_model(model_location):
    return tf.keras.models.load_model(model_location)


def cv2_to_tensor(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 160))
    image_array = np.asarray(image)
    return np.expand_dims(image_array, axis=0)


def predict(model, frame):
    tensor = cv2_to_tensor(frame)
    predictions = model.predict(tensor).flatten()
    predictions = tf.nn.sigmoid(predictions)

    return predictions


def stream_and_predict(model, video):
    while True:
        ret, frame = video.read()
        prediction = predict(model, frame).numpy()

        cv2.putText(frame, str(prediction), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q') & 0xFF:
            break


def main():
    model = load_model('layered_model')
    vid = cv2.VideoCapture(0)
    stream_and_predict(model, vid)
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
