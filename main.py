import tensorflow as tf
from transformers import pipeline
from googletrans import Translator
import numpy as np
import re
import speech_recognition as sr
import cv2

# Function for Real-Time Feedback on Communication
def analyze_communication(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    text = translation.text  # Translate to English if necessary

    # Analyze for filler words
    filler_words = ["um", "uh", "like", "you know", "basically", "actually"]
    filler_count = sum([len(re.findall(rf"\\b{word}\\b", text.lower())) for word in filler_words])

    # Count pauses (long periods with no words)
    pauses = len(re.findall(r"\.\.\.", text))

    return {
        "filler_count": filler_count,
        "pauses": pauses,
        "translated_text": text
    }

# Function for Sentiment Analysis using Hugging Face pipeline
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    sentiment = sentiment_pipeline(text)

    return {
        "label": sentiment[0]["label"],
        "score": sentiment[0]["score"]
    }

# Function for Soft Skill Evaluation (Tone and Emotion)
def evaluate_soft_skills(text):
    tone_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")
    tones = tone_pipeline(text)

    # Aggregate scores for the top 3 emotions
    sorted_tones = sorted(tones, key=lambda x: x['score'], reverse=True)[:3]

    return {tone["label"]: tone["score"] for tone in sorted_tones}

# Function to process live audio and video
def process_live_audio_video():
    # Initialize the video capture and speech recognizer
    video_capture = cv2.VideoCapture(0)
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Starting live audio and video processing. Press 'q' to quit.")

    while True:
        # Capture video frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Display the video frame
        cv2.imshow('Live Video', frame)

        # Process audio
        with microphone as source:
            print("Listening for audio...")
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                print("Recognized Text:", text)

                # Perform NLP analysis
                communication_feedback = analyze_communication(text)
                sentiment_feedback = analyze_sentiment(text)
                soft_skill_feedback = evaluate_soft_skills(text)

                print("Communication Feedback:", communication_feedback)
                print("Sentiment Feedback:", sentiment_feedback)
                print("Soft Skill Feedback:", soft_skill_feedback)

            except sr.WaitTimeoutError:
                print("No speech detected.")
            except sr.UnknownValueError:
                print("Speech was unintelligible.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    video_capture.release()
    cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    process_live_audio_video()
