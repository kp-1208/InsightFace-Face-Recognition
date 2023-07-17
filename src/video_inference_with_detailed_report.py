import sys
import time
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')

from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from random import randint
from imutils import paths
import face_preprocess
import pandas as pd
import numpy as np
import face_model
import argparse
import pickle
import dlib
import cv2
import os

start = time.time()

ap = argparse.ArgumentParser()

ap.add_argument("--mymodel", default="outputs/my_model.h5",
    help="Path to recognizer model")
ap.add_argument("--le", default="outputs/le.pickle",
    help="Path to label encoder")
ap.add_argument("--embeddings", default="outputs/embeddings.pickle",
    help='Path to embeddings')
ap.add_argument("--video-out", default="../datasets/videos_output/video_test.mp4",
    help='Path to output video')
ap.add_argument("--video-in", default="../datasets/videos_input/test.mp4")


ap.add_argument('--image-size', default='112,112', help='')
ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
ap.add_argument('--ga-model', default='', help='path to load model.')
ap.add_argument('--gpu', default=0, type=int, help='gpu id')
ap.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

args = ap.parse_args()

# Load embeddings and labels
data = pickle.loads(open(args.embeddings, "rb").read())
le = pickle.loads(open(args.le, "rb").read())

#cols = ["name", "id", "Time", "Date", "max_distance", "min_distance", "max_prob", "min_prob", "cos_range", "prob_range"]
#print(type(cols))
df_inferred_faces = pd.DataFrame(columns=['name', 'id', 'time', 'date', 'max_distance', 'min_distance', 'max_prob', 'min_prob', 'cos_range', 'prob_range'],dtype=np.float64)

print(type(df_inferred_faces))

embeddings = np.array(data['embeddings'])
labels = le.fit_transform(data['names'])

# Initialize detector
detector = MTCNN()

# Initialize faces embedding model
embedding_model =face_model.FaceModel(args)

# Load the classifier model
model = load_model(args.mymodel)

def store_inferred_face_in_dataframe(name_of_person: str, cos_distance: float, proba: float):
    '''
        This function will store the name of the person and the face distance in the dataframe

        Arguments:
            name_of_person {string} -- name of the person
            face_distance {float} -- face distance
            cam_name {string} -- name of the camera
            cam_ip {string} -- ip address of the camera

        Returns:
            None
    '''

    # current time in HH:MM:SS format
    current_time = time.strftime("%H:%M:%S")
    # current date in DD/MM/YYYY format
    current_date = time.strftime("%d/%m/%Y")

    # The name of person is in the format <name>_<id>
    # Split name_of_person into name and id if there is an underscore in the name
    # if '_' in name_of_person:
    #     name, id = name_of_person.rsplit('_', maxsplit=1)
    # else:
    name = name_of_person
    # Give a random id to the person
    id = f'unknown_{randint(0, 1000000)}'

    # We want to store the details of the person only if it is not already present in the dataframe
    if name not in df_inferred_faces['name'].values:
    
        # store the name of the person in the dataframe
        df_inferred_faces.loc[len(df_inferred_faces)] = [name, id, current_time, current_date, cos_distance, cos_distance, proba, proba, 0, 0]

        # Uncomment the following line to display the details of the person recognized on console
        print(f'Inferred face: {name} ID: {id} at time: {current_time} on date: {current_date}')
        
        
    # Update the min, max face distance and proba of existing records
    else:
        print(f'Inferred face: {name} ID: {id} at time: {current_time} on date: {current_date}')
        if float(df_inferred_faces[df_inferred_faces.name == name].max_distance) < cos_distance:
            index = df_inferred_faces[df_inferred_faces.name == name].index[0]
            df_inferred_faces.loc[index, 'max_distance'] = np.float64(cos_distance)
            
        elif float(df_inferred_faces[df_inferred_faces.name == name].min_distance) > cos_distance:
            index = df_inferred_faces[df_inferred_faces.name == name].index[0]
            df_inferred_faces.loc[index, 'min_distance'] = np.float64(cos_distance)
            
        df_inferred_faces['cos_range'] = df_inferred_faces['max_distance'] - df_inferred_faces['min_distance']
        
        if float(df_inferred_faces[df_inferred_faces.name == name].max_prob) < proba:
            index = df_inferred_faces[df_inferred_faces.name == name].index[0]
            df_inferred_faces.loc[index, 'max_prob'] = np.float64(proba)
            
        elif float(df_inferred_faces[df_inferred_faces.name == name].min_prob) > proba:
            index = df_inferred_faces[df_inferred_faces.name == name].index[0]
            df_inferred_faces.loc[index, 'min_prob'] = np.float64(proba)
            
        df_inferred_faces['prob_range'] = df_inferred_faces['max_prob'] - df_inferred_faces['min_prob']
        
        '''if float(df_inferred_faces[df_inferred_faces.name == name].distance_range) > RANGE_TOLERANCE:
            if name not in df_final['name'].values:
                df_final.loc[len(df_final)] = [name, id, current_time, current_date, cam_name, cam_ip]'''

def store_dataframe_in_csv():
    '''
        This function will store the dataframe in a csv file

        Arguments:
            None

        Returns:
            True if the dataframe is stored in the csv file, False otherwise
    '''
    global df_inferred_faces

    try:

        # remove the last element from the string as it is the file name
        #report_path_dir = REPORT_PATH.rsplit('/', maxsplit=1)[0]
        # create a directory for logs if it does not exist
        #check_for_directory(report_path_dir)
        #dfmax = df_inferred_faces.groupby(['name'], as_index = False).max()
        #dfmin = df_inferred_faces.groupby(['name'], as_index = False).min()
        #dfmax.rename(columns={'face distance':'max_distance'}, inplace = True)
        #dfmin.rename(columns={'face distance':'min_distance'}, inplace = True)
        #dfmax = (dfmax.merge(dfmin, left_on='name', right_on='name').reindex(columns=['name', 'max_distance', 'min_distance']))
        #dfmax['range'] = dfmax['max_distance'] - dfmax['min_distance']
        #df_final = dfmax[dfmax.range>0.022]
        #del dfmax, dfmin
        df_inferred_faces.to_csv("detailed_report.csv", encoding='utf-8')
        #logger.info('Dataframe stored in csv file')
        return True
    except Exception as e:
        return False
    finally:
        # reset the dataframe
        # df_inferred_faces = pd.DataFrame(columns=['name', 'id', 'time', 'date', 'cam name', 'cam ip', 'face distance'])
        pass





# Define distance function
def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

# Initialize some useful arguments
cosine_threshold = 0.9
proba_threshold = 0.85
comparing_num = 5
trackers = []
texts = []
frames = 0

# Start streaming and recording
cap = cv2.VideoCapture(args.video_in)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_width = 800
save_height = int(800/frame_width*frame_height)
video_out = cv2.VideoWriter(args.video_out, cv2.VideoWriter_fourcc('M','J','P','G'), 24, (save_width,save_height))

while True:
    try:
        ret, frame = cap.read()
        frames += 1
        frame = cv2.resize(frame, (save_width, save_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frames%3 == 0:
            trackers = []
            texts = []

            detect_tick = time.time()
            bboxes = detector.detect_faces(frame)
            detect_tock = time.time()

            if len(bboxes) != 0:
                reco_tick = time.time()
                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0], landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                     landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1], landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2,5)).T
                    nimg = face_preprocess.preprocess(frame, bbox, landmarks, image_size='112,112')
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2,0,1))
                    embedding = embedding_model.get_feature(nimg).reshape(1,-1)

                    text = "Unknown"

                    # Predict class
                    preds = model.predict(embedding)
                    preds = preds.flatten()
                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity = CosineSimilarity(embedding, compare_embeddings)
                    if cos_similarity < cosine_threshold and proba > proba_threshold:
                        name = le.classes_[j]
                        text = "{}".format(name)
                        store_inferred_face_in_dataframe(text, cos_similarity, proba)
                        print("Recognized: {} <{:.2f}>".format(name, proba*100))
                    # Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    texts.append(text)
    
                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255,0,0), 2)
        else:
            for tracker, text in zip(trackers,texts):
                pos = tracker.get_position()
    
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
    
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255),2)
    
        cv2.imshow("Frame", frame)
        video_out.write(frame)
        # print("Faces detection time: {}s".format(detect_tock-detect_tick))
        # print("Faces recognition time: {}s".format(reco_tock-reco_tick))
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
       
    except:
        print("Complete processing")
        store_dataframe_in_csv()
        break
        
video_out.release()
cap.release()
cv2.destroyAllWindows()
end = time.time()
print("Execution time: {:.2f} seconds".format(end - start))

