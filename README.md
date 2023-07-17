# InsightFace-Face-Recognition
This project is to create an ML Model to recognize faces in images, videos, and live stream on CPU as well as GPU.

## Using the model

### Step 1:
Load the dataset as separate folders for different identities in the __/datasets/train__ directory. It will be in the following structure:
```
/datasets
  /train
    /person1
      + face_01.jpg
      + face_02.jpg
      + ...
    /person2
      + face_01.jpg
      + face_02.jpg
      + ...
    / ...
  /test
  /unlabeled_faces
  /videos_input
  /videos_output
```

### Step 2:
Provide the path of the dataset in the path argument specified in files in __/src__.
**In preprocess_dataset.py:**
```python
# Add the path to dataset in imagePaths.

start = time.time()

detector = MTCNN()

imagePaths = list(paths.list_images("/home/kp1208/Desktop/InsightFace-Face-Recognition/datasets/train"))
print(imagePaths)
```

**In delete_wrong_files.py:**
```python
start = time.time()

detector = MTCNN()

target_resolution = 37632  # Provide your desired resolution here
imagePaths = list(paths.list_images("/home/kp1208/Desktop/InsightFace-Face-Recognition/datasets/train"))
print(imagePaths)
```

### Step 3:
Run the __/preprocess_and_train.sh__ script using the following command in __/src__.
```bash
. preprocess_and_train.sh
```
The script mentioned above executes 4 python scripts sequentially:
```
#!/bin/bash

start_time=$(date +%s.%N)

python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/preprocess_dataset.py
python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/delete_wrong_files.py
python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/faces_embedding.py
python /home/kp1208/Desktop/InsightFace-Face-Recognition/src/train_softmax.py

end_time=$(date +%s.%N)
execution_time=$(echo "$end_time - $start_time" | bc)

echo "Total Preprocessing and Training Time: $execution_time seconds"
```

### Step 4:
Now, you have the trained classifier as well as the face embeddings. Use them to inference on your custom data.

**To inference on an image:**
```bash
python insightface_image_recognition.py
```

**To inference on a video:**
```bash
python insightface_video_recognition.py
```

**To inference on a video and generate detailed report with cos_similarity and probability:**
```bash
python video_inference_with_detailed_report.py
```

**All the reports will be saved in __/src__ directory.**
