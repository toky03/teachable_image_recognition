# Live Prediction
Use a tensorflow Model to create a live prediction from a Webcam

## Requirements
1. Save a Tensorflow Model into this Project directory Named `layered_model`
2. The Model needs to have a Input Shape of `(160, 160)`
3. The Output needs to be a single number

## Run the Model
1. (Optional) create a virtual environemnt `python3 -m venv venv`
2. (Optional) virtual environemnt already exists start it `source venv/bin/activate`
3. Install all the dependencies stated in [requirements.txt](./requirements.txt) `pip install requirements.txt`
4. Start the live Prediction `python3 main.py`
5. Quit the prediction with pressing key `q`
