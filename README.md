# Image Recognition

Collect, train and predict with a Tensorflow Machine Learning Model

The Project contains a subfolder for each step
- [photo_collector](./photo_collector) to take images separated per label
- [model_trainer](./model_trainer) to build train and validate a tensorflow based on images collected from [photo_collector](./photo_collecor)
- [live_prediction](./live_prediction) uses the model created in [model_trainer](./model_trainer) to show the prediction live from the webcam
- edge_live_prediction same as live_prediction but for edge devices (Raspberry Pi) and in Headless mode
