## Installation

```bash
# create python venv
python -m venv .env 
# activate venv
source .env/bin/activate
# install dependencies
python -m pip install -r requirements.txt
# download the models & test videos
./setup.sh
```

```bash
python -m pip install git+https://github.com/roboflow/sports.git
```

## Usage

```bash
# generate an occluded video
python occlude_video.py videos/video_1.mp4 videos/occlude_1.mp4 -x 500 -W 100
# run the MOT algorithm
# use '-d cpu' instead of '-d cuda' if no nvidia GPU
python main.py -i videos/occlude_1.mp4 -o logs/video_1.mp4 -d cuda -f 100
```

## TODO

- Get a dataset to train on.
- Figure out how to identify bad frames (i.e., no useful information); use detection confidence?
- Figure out better way to handle homography noise.