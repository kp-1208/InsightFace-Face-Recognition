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

### Step 3:
Run the __/preprocess_and_train.sh__ script using the following command in __/src__.
```
. preprocess_and_train.sh
```
The script mentioned above executes 4 python scripts sequentially:
__/preprocess_dataset.py__ -> __/delete_wrong_files.py__ -> __/faces_embedding.py__ -> __/train_softmax.py__

### Step 4:
Now, you have the trained classifier as well as the face embeddings. Use them to inference on your custom data.

**To inference on an image:**
```
python insightface_image_recognition.py
```

**To inference on a video:**
```
python insightface_video_recognition.py
```

**To inference on a video and generate detailed report with cos_similarity and probability:**
```
python video_inference_with_detailed_report.py
```

**All the reports will be saved in __/src__ directory.**
