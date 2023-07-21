# Photo Collector
Collect labeled Images to train a Tensorflow model

## Run Photo Collector
1. (Optional) create a virtual environemnt `python3 -m venv venv`
2. (Optional) virtual environemnt already exists start it `source venv/bin/activate`
3. Install all the dependencies stated in [requirements.txt](./requirements.txt) `pip install -r requirements.txt`
4. Start webcam `python3 main.py`
5. Take photos from webcam with key `a`, `s`, `d` and `l` to label them
6. The Images are saved inside the folder `image_set`