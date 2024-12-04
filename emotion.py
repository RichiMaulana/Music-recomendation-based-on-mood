import cv2
from deepface import DeepFace
from tkinter import Listbox, Tk, Label, Button, END as tkEND  # noqa: F403
from PIL import Image, ImageTk
import pandas as pd

songs = pd.read_csv("data/song.csv")

# Initialize the main Tkinter window
window = Tk()
window.title("Emotion Detection App")
window.geometry("1000x800")

# Initialize the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Initialize the webcam
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

# Define the label to display video feed
video_label = Label(window)
video_label.pack(pady=(0, 20))

# Label to display additional information
info_label = Label(window, text="Press Capture to analyze emotions", font=("Helvetica", 16))
info_label.pack(pady=(0, 20))

def update_frame():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        return
    
    # Convert the frame to an ImageTk format for Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    
    # Continue updating frames
    video_label.after(10, update_frame)

def capture_and_analyze():
    # Capture a single frame from the webcam
    ret, frame = cap.read()
    if not ret:
        info_label.config(text="Failed to capture image")
        return
    
    # Convert to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are detected, process each face
    dominant_emotion = ""
    for (x, y, w, h) in faces:
        # Extract the face area from the frame
        face = frame[y:y+h, x:x+w]
        
        # Use DeepFace to analyze emotions on the face
        result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
        
        # Get the dominant emotion and add to the list
        dominant_emotion = result[0]['dominant_emotion']
        
        # Draw rectangle and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Update the info label with detected emotions
    info_text = "Detected Emotions: " + dominant_emotion if dominant_emotion else "No faces detected"
    info_label.config(text=info_text)
    
    # Display the captured frame with annotations in Tkinter
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # # display the song rec
    if dominant_emotion in ['sad', 'neutral', 'angry', 'happy']:

        song_index = 0
        song_recommend = songs[songs["mood"] == dominant_emotion].sample(n=10)
        for row in song_recommend.itertuples():
            song_list.delete(song_index)
            song_list.insert(song_index, row[14])
            song_index +=1
        
        song_list.pack()
        song_index = 0

# Button to capture and analyze emotions
capture_button = Button(window, text="Capture", command=capture_and_analyze, font=("Helvetica", 14))
capture_button.pack(pady=(0, 20))

rec_label = Label(window, text="Here is the song recommendation for you: ", font=("Helvetica", 16))
rec_label.pack(pady=(0, 5))  


song_list = Listbox(window, height=10, width=50, activestyle=["dotbox"])

for i in range(10):
    song_list.insert(tkEND, "-")
song_list.pack(pady=(0, 20))
# song_list.activate()

# Start video stream in Tkinter window
update_frame()

# Run the Tkinter main loop
window.mainloop()

# Release the webcam resource after closing the window
cap.release()
