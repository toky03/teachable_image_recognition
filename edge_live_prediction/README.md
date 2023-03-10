# Live Prediction in headless mode (Led indication)
Use a tensorflow Model to create a live prediction from a Webcam indicated with a led


# Install requirements:
sudo apt install python3-opencv

## Requirements
1. Save a Tensorflow Model into this Project directory Named `layered_model` or run [model_trainer](../model_trainer/)
2. The Model needs to have a Input Shape of `(160, 160)`
3. The Output needs to be a single number

## Run the Model
1. Install tensorflow Lite
  a. either from [source](https://www.tensorflow.org/lite/guide/build_cmake_pip)
  b. for Raspberry Pi or Coral Devices: with [tflite-runtime](https://www.tensorflow.org/lite/guide/python)
  c. directly with `python3 -m pip install tflite-runtime`
2. Install Opencv `sudo apt install python3-opencv`
3. Start the live Prediction `python3 main.py`
4. As per version 4.5 from opencv on a raspberry pi leagacy camera support needs to be enabled see [raspberry forum](https://forums.raspberrypi.com/viewtopic.php?t=327192)
5. Quit the prediction with pressing key `q`