import pickle
import pandas as pd
import mediapipe as mp
import cv2
import numpy as np

with open('deadlift.pkl', 'rb') as f:
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

counter = 0

cap = cv2.VideoCapture(1)
#initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, image = cap.read()
        
        #Recolor Feed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        #Make Detections
        results = pose.process(image)
        
        #Recolor image back to BGR for rendering
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,177,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_langugage_class = model.predict(X)[0]
            body_langugage_prob = model.predict_proba(X)[0]
            print(body_langugage_class, body_langugage_prob)

            if body_langugage_class == 'down' and body_langugage_class == 'down' and body_langugage_prob[body_langugage_class.argmax()] >= .7:
                current_stage = 'down'
            elif current_stage == 'down' and body_langugage_class == body_langugage_class == 'up' and body_langugage_prob[body_langugage_class.argmax()] >= .7:
                current_stage = 'up'
                counter += 1
                print(current_stage)

            #status box
            cv2.rectangle(image, (0,0), (250,60), (245, 177, 16), -1)

            #display class
            cv2.putText(image, 'CLASS'
                            , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, body_langugage_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            #display probability
            cv2.putText(image, 'PROB'
                            , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_langugage_prob[np.argmax(body_langugage_class)]))
                            , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)     

            #display count
            cv2.putText(image, 'COUNT'
                            , (180,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter)
                            , (175,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)       
        
        except Exception as e:
            pass
        cv2.imshow('Raw Webcam Feed', image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()