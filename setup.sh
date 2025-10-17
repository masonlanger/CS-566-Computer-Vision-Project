#!/bin/bash

# Get the directory where the script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source "$DIR/.env/bin/activate"
if [[ ! -e $DIR/models ]]; then
    mkdir "$DIR/models"
else
    echo "'models' directory already exists."
fi

# download the models
gdown -O "$DIR/models/football-ball-detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
gdown -O "$DIR/models/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
gdown -O "$DIR/models/football-pitch-detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

# download the videos
gdown -O "$DIR/models/video_1.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
gdown -O "$DIR/models/video_2.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
gdown -O "$DIR/models/video_3.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
gdown -O "$DIR/models/video_4.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
gdown -O "$DIR/models/video_5.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"
