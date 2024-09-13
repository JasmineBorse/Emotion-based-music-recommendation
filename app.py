# Importing modules
import base64
import numpy as np
import streamlit as st
import cv2
import pandas as pd
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load data
df = pd.read_csv("/Users/jasmineborse/Desktop/Projects/Emotion-based-music-recommendation/muse_v3.csv")
df['link'] = df['lastfm_url']
df['name'] = df['track']
df['emotional'] = df['number_of_emotion_tags']
df['pleasant'] = df['valence_tags']
df = df[['name', 'emotional', 'pleasant', 'link', 'artist']]
df = df.sort_values(by=["emotional", "pleasant"])
df.reset_index(drop=True, inplace=True)

# Split dataset based on emotions
df_sad = df[:18000]
df_fear = df[18000:36000]
df_angry = df[36000:54000]
df_neutral = df[54000:72000]
df_happy = df[72000:]

def fun(list):
    data = pd.DataFrame()
    if len(list) == 1:
        v = list[0]
        t = 30
        if v == 'Neutral':
            data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
        elif v == 'Angry':
            data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
        elif v == 'Fear':
            data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
        elif v == 'Happy':
            data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
        else:
            data = pd.concat([data, df_sad.sample(n=t)], ignore_index=True)
    elif len(list) == 2:
        times = [30, 20]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)], ignore_index=True)
    elif len(list) == 3:
        times = [55, 20, 15]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)], ignore_index=True)
    elif len(list) == 4:
        times = [30, 29, 18, 9]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)], ignore_index=True)
    else:
        times = [10, 7, 6, 5, 2]
        for i in range(len(list)):
            v = list[i]
            t = times[i]
            if v == 'Neutral':
                data = pd.concat([data, df_neutral.sample(n=t)], ignore_index=True)
            elif v == 'Angry':
                data = pd.concat([data, df_angry.sample(n=t)], ignore_index=True)
            elif v == 'Fear':
                data = pd.concat([data, df_fear.sample(n=t)], ignore_index=True)
            elif v == 'Happy':
                data = pd.concat([data, df_happy.sample(n=t)], ignore_index=True)
            else:
                data = pd.concat([df_sad.sample(n=t)], ignore_index=True)
    return data

def pre(l):
    emotion_counts = Counter(l)
    result = []
    for emotion, count in emotion_counts.items():
        result.extend([emotion] * count)
    ul = []
    for x in result:
        if x not in ul:
            ul.append(x)
    return ul



# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights("/Users/jasmineborse/Desktop/Projects/Emotion-based-music-recommendation/model.h5")


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Function to perform emotion detection
def detect_emotion(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    emotions_detected = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        max_index = int(np.argmax(prediction))
        emotions_detected.append(emotion_dict[max_index])

    return emotions_detected

#streamlit

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

image_base64 = image_to_base64("bg.jpg")

page_bg_img = f'''
<style>
body {{
    background-image: url("data:image/jpeg;base64,{image_base64}");
    background-size: cover;
}}
</style>
'''


st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: white'><b>Emotion Based Music Recommendation</b></h2>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: grey;'><b>Select emotions to get music recommendations</b></h5>", unsafe_allow_html=True)

# Option to upload image for emotion detection
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
# # Add this to the sidebar for webcam capture
webcam_option = st.sidebar.checkbox("Use Webcam")

if webcam_option:
    st.sidebar.write("Click the button below to start the webcam.")
    camera_image = st.camera_input("Capture Image")

    if camera_image is not None:
        # Convert the captured image to a format that OpenCV can handle
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        
        if image is not None:
            detected_emotions = detect_emotion(image)
            st.write("Detected Emotions: ", detected_emotions)
        else:
            st.error("Error: Could not decode the image from webcam.")


if st.sidebar.button('Get Recommendations'):
    if webcam_option and camera_image is not None:
        # Process captured image
        image = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
        detected_emotions = detect_emotion(image)
        st.write("Detected Emotions: ", detected_emotions)
    elif uploaded_file is not None:
        # Process uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        detected_emotions = detect_emotion(image)
        st.write("Detected Emotions: ", detected_emotions)
    
    if detected_emotions:
        # Get music recommendations
        
        new_df = fun(detected_emotions)
        st.write("")
        st.markdown("<h5 style='text-align: center; color: grey;'><b>Recommended Songs with Artist Names</b></h5>", unsafe_allow_html=True)
        st.write("")
        if not new_df.empty:
            for index, row in new_df.iterrows():
             st.write(f"**{row['name']}** by {row['artist']}")
             st.markdown(f"[Listen Here]({row['link']})")
        else:
            st.write("No recommendations available for the detected emotions.")
else:
    st.warning("Please upload an image or use the webcam to detect emotions and get recommendations.")


# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: white;'>Created by Jasmine Borse.</h6>", unsafe_allow_html=True)

