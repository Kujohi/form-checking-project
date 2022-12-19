import tkinter as tk 
import customtkinter as ck 

import pandas as pd 
import numpy as np 
import pickle 

import mediapipe as mp
import cv2
from PIL import Image, ImageTk 

from landmarks import landmarks
import math

window = tk.Tk()
window.geometry("718x541")
window.title("GYM OR SOMETHING????")
window.configure(bg="white")
ck.set_appearance_mode("Dark")

#Output

frame1 = tk.Frame(height=360,width=238,bg="#F3CCFF")
frame1.place(x=479, y=92)

classLabel = ck.CTkLabel(window, height=40, width=120, text_color="#A555EC",bg_color="#F3CCFF",font=("Arial", 15,"bold"))
classLabel.place(x=540, y=126)
classLabel.configure(text='STAGE') 

counterLabel = ck.CTkLabel(window, height=40, width=120, text_color="#A555EC",bg_color="#F3CCFF",font=("Arial", 15,"bold"))
counterLabel.place(x=540, y=227)
counterLabel.configure(text='REPS') 

probLabel  = ck.CTkLabel(window, height=40, width=120, text_color="#A555EC",bg_color="#F3CCFF",font=("Arial", 15,"bold"))
probLabel.place(x=540, y=328)
probLabel.configure(text='PROB') 

classBox = ck.CTkLabel(window, height=46, width=117, text_color="white", fg_color="black",font=("Arial", 30,"bold"))
classBox.place(x=540, y=160)
classBox.configure(text='0') 

counterBox = ck.CTkLabel(window, height=46, width=117, text_color="white", fg_color="black",font=("Arial", 30,"bold"))
counterBox.place(x=540, y=261)
counterBox.configure(text='0') 

probBox = ck.CTkLabel(window, height=46, width=117, text_color="white", fg_color="black",font=("Arial", 30,"bold"))
probBox.place(x=540, y=362)
probBox.configure(text='0') 

#Day - decorate

frame2 = tk.Frame(height=92,width=255,bg="#A555EC")
frame2.place(x=464, y=0)
dayLabel = ck.CTkLabel(frame2, height=40, width=120, text_color="white",font=("Arial", 30,"bold"))
dayLabel.place(x=70,y=25)
dayLabel.configure(text='Day1')

#Type
typeLabel = ck.CTkLabel(window, height=40, width=120, text_color="#CB3BB4",font=("Arial", 30,"bold"))
typeLabel.place(x=340,y=22)
typeLabel.configure(text='SIT UP')

#User
ima = Image.open("./img/user2.webp")
ima = ima.resize((49,49), Image.ANTIALIAS)
test = ImageTk.PhotoImage(ima)

imageLabel = tk.Label(image=test)
imageLabel.image = test
imageLabel.place(x=30, y=15)

#Name 
typeLabel = ck.CTkLabel(window, height=40, width=120, text_color="#A555EC",font=("Arial", 20,"bold"))
typeLabel.place(x=85,y=22)
typeLabel.configure(text='USERNAME')



def reset_counter(): 
    global counter
    counter = 0 


#Buttons
resetButton = ck.CTkButton(window, text='RESET', command=reset_counter, height=50, width=126, text_color="white", fg_color="#A555EC",corner_radius=20)
resetButton.place(x=296, y=471)


frame = tk.Frame(height=360,width=480,bg="#F3CCFF")
frame.place(x=0, y=92) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) 

with open('deadlift.pkl', 'rb') as f: 
    model = pickle.load(f) 

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 

def detect(): 
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob 

    ret, frame = cap.read()
    frame = cv2.resize(frame, (480,360))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(image)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,177,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )

    try: 
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks) 
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0] 

        if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
            current_stage = "down" 
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
            current_stage = "up" 
            counter += 1 

    except Exception as e: 
        print(e) 

    img = image[:, :460, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(1, detect)  

    counterBox.configure(text=counter) 
    probBox.configure(text=round(bodylang_prob[bodylang_prob.argmax()],2)) 
    classBox.configure(text=current_stage) 

detect() 
window.mainloop()