# Live Prediction
Use a tensorflow Model to create a live prediction from a Webcam

## Requirements
1. Save a Tensorflow classification Model into this Project directory Named `classification_model/layered_model_edgetpu.tflite` or run [model_trainer](../model_trainer/)
2. Download an Object Detection Model from [coral](https://coral.ai/models/object-detection/)
2. The Model needs to have a Input Shape of `(640, 640)`
3. The Output needs to be a single number

## Run the Model
1. (Optional) create a virtual environemnt `python3 -m venv venv`
2. (Optional) virtual environemnt already exists start it `source venv/bin/activate`
3. Install all the dependencies stated in [requirements.txt](./requirements.txt) `pip install -r requirements.txt`
4. Install tensorflow Lite
  a. either from [source](https://www.tensorflow.org/lite/guide/build_cmake_pip)
  b. for Raspberry Pi or Coral Devices: with [tflite-runtime](https://www.tensorflow.org/lite/guide/python)
  c. For coral install all the required dependencies from [coral.ai](https://coral.ai/software/#pycoral-api)
5. Start the live Prediction `python3 main.py`
6. Quit the prediction with pressing key `q`
