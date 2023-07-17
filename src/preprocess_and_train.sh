#!/bin/bash

start_time=$(date +%s.%N)

python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/preprocess_dataset.py
python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/delete_wrong_files.py
python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/faces_embedding.py
python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/train_softmax.py

end_time=$(date +%s.%N)
execution_time=$(echo "$end_time - $start_time" | bc)

echo "Total Preprocessing and Training Time: $execution_time seconds"


