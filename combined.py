import cv2
import numpy as np
from keras.models import load_model
import pickle
import face_recognition

# load mask detection model
model = load_model("./model2-018.model")
labels_dict = {0: 'without mask', 1: 'mask'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
size = 4

# load face detection model
f = open("ref_name.pkl", "rb")
ref_dictt = pickle.load(f)  # ref_dict=ref vs name
f.close()

f = open("ref_embed.pkl", "rb")
embed_dictt = pickle.load(f)  # embed_dict- ref  vs embedding
f.close()
known_face_encodings = []  # encodingd of faces
known_face_names = []  # ref_id of faces
for ref_id, embed_list in embed_dictt.items():
	for embed in embed_list:
		known_face_encodings += [embed]
		known_face_names += [ref_id]

# Initialize some variables for face detection


webcam = cv2.VideoCapture(0)  # Use camera 0

# We load the xml file
classifier = cv2.CascadeClassifier(
	cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# classifier = cv2.CascadeClassifier('/home/ASUS/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')

#while True:

def camera_stream():  
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    (rval, im) = webcam.read()
    im = cv2.flip(im, 1, 1)  # Flip to act as a mirror

    frame = im.copy()

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces = classifier.detectMultiScale(mini)

    print(len(faces))
    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f]  # Scale the shapesize backup
        # Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (150, 150))
        normalized = resized/255.0
        reshaped = np.reshape(normalized, (1, 150, 150, 3))
        reshaped = np.vstack([reshaped])
        result = model.predict(reshaped)
        print("here")
        print(result)

        label = np.argmax(result, axis=1)[0]

        # if mask found:
        if label==1:
            print("mask found!")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            

        else: # mask not found, do face detect
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    # else:
                    #     return cv2.imencode('.jpg', frame)[1].tobytes()
                    #     continue
                    face_names.append(name)

            process_this_frame = not process_this_frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # updating in database

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35),
                            (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                if name == 'Unknown':
                    cv2.putText(frame, 'Face not recognized', (left + 6,
                            bottom - 6), font, 1.0, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, ref_dictt[name], (left + 6,
                            bottom - 6), font, 1.0, (255, 255, 255), 1)
                im = frame
        
        return cv2.imencode('.jpg', frame)[1].tobytes()

#     # Show the image
#     cv2.imshow('LIVE',   im)
#     key = cv2.waitKey(10)
#     # if Esc key is press then break out of the loop
#     if key == 27:  # The Esc key
#         break
# # Stop video
# webcam.release()

# # Close all started windows
# cv2.destroyAllWindows()
