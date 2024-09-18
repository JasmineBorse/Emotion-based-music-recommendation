

# Emotion-Based Music Recommendation System

## About the Project

The Emotion-Based Music Recommendation System is an interactive application that provides personalized music recommendations based on the emotional content of an image. By leveraging computer vision and machine learning, this system analyzes images uploaded by users or captured via webcam to detect emotions and suggest songs that align with those emotions.

## Features

- **Emotion Detection:** Uses a Convolutional Neural Network (CNN) to detect emotions from facial images.
- **Music Recommendations:** Recommends songs based on detected emotions from a  dataset.
- **User Interface:** Built with Streamlit for an intuitive and interactive web experience.

## Components

1. **Data Preparation:**
   - Dataset: `muse_v3.csv` containing song information and emotional tags.
   
2. **Machine Learning Model:**
   - **Architecture:** CNN with convolutional layers, max-pooling layers, dropout layers, and dense layers.
   - **Training:** Model trained on facial emotion data and saved as `model.h5`.

3. **Emotion Detection:**
   - **Face Detection:** Utilizes OpenCV's Haar Cascade Classifier to detect faces in images.
   - **Emotion Classification:** Processes detected faces with the CNN model to predict emotions.

4. **Recommendation System:**
   - **Sampling:** Retrieves and displays songs matching detected emotions from the dataset.
   - **Display:** Shows song name, artist, and listening link.

5. **User Interface:**
   - **Streamlit Application:** Allows users to upload images or use a webcam for emotion detection and music recommendations.
   

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/JasmineBorse/emotion-based-music-recommendation.git
   cd emotion-based-music-recommendation
   ```

2. **Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model:**

   Ensure `model.h5` is placed in the project directory.

## Usage

1. **Run the Streamlit App:**

   ```bash
   streamlit run app.py
   ```

2. **Interact with the App:**
   - Upload an image or use the webcam to capture your facial expression.
   - Click "Get Recommendations" to receive personalized music suggestions based on detected emotions.

## Requirements

- Python 3.x
- Streamlit
- TensorFlow/Keras
- OpenCV
- Pandas
- Numpy

Refer to `requirements.txt` for a complete list of dependencies.

##Output Images
Have a look at the project here:

[PDF](https://github.com/JasmineBorse/emotion-based-music-recommendation/raw/main/Output_pdf_emotion_based_music_rec.pdf)




## Contributing

If you would like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io/) for providing the web framework.
- [OpenCV](https://opencv.org/) for face detection.
- [TensorFlow/Keras](https://www.tensorflow.org/) for the machine learning model.

