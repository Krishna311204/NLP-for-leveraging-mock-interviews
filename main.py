import tensorflow as tf
from transformers import pipeline
from googletrans import Translator
import numpy as np
import re
import speech_recognition as sr
import cv2
from deepface import DeepFace

# Global counters for analysis
total_filler_words = 0
total_pauses = 0
emotion_scores = []
sentiment_scores = []

# Function for Real-Time Feedback on Communication
def analyze_communication(text):
    global total_filler_words, total_pauses

    translator = Translator()
    translation = translator.translate(text, dest='en')
    text = translation.text  # Translate to English if necessary

    # Analyze for filler words
    filler_words = ["um", "uh", "like", "you know", "basically", "actually"]
    filler_count = sum([len(re.findall(rf"\b{word}\b", text.lower())) for word in filler_words])
    total_filler_words += filler_count  # Update global counter

    # Count pauses (long periods with no words)
    pauses = len(re.findall(r"\.\.\.", text))
    total_pauses += pauses  # Update global counter

    return {
        "filler_count": filler_count,
        "pauses": pauses,
        "translated_text": text
    }

# Function for Sentiment Analysis using Hugging Face pipeline
def analyze_sentiment(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased")
    sentiment = sentiment_pipeline(text)

    sentiment_scores.append(sentiment[0]["score"])  # Track sentiment score
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

# Function to analyze facial expressions
def analyze_face_expression(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result and isinstance(result, list) and len(result) > 0:
            dominant_emotion = result[0]['dominant_emotion']
            emotion_scores.append(dominant_emotion)  # Track emotions
            return dominant_emotion
        return "No face detected"
    except Exception as e:
        print(f"Error in facial analysis: {e}")
        return "Error"

# Function to calculate the final confidence score
def calculate_confidence():
    # Base confidence is 5, penalties are subtracted based on detected issues
    confidence = 5.0  

    # Reduce confidence for excessive filler words
    if total_filler_words > 10:
        confidence -= 1
    elif total_filler_words > 20:
        confidence -= 2

    # Reduce confidence for excessive pauses
    if total_pauses > 5:
        confidence -= 1
    elif total_pauses > 10:
        confidence -= 2

    # Consider sentiment scores (higher is better)
    avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
    if avg_sentiment < 0.6:
        confidence -= 1

    # Consider facial emotions (reduce for negative emotions)
    negative_emotions = ["angry", "sad", "fear"]
    negative_count = sum(1 for e in emotion_scores if e in negative_emotions)
    if negative_count > 3:
        confidence -= 1

    # Ensure confidence is within bounds (0 to 5)
    return max(0, min(5, confidence))

# Function to process live audio and video
def process_live_audio_video():
    video_capture = cv2.VideoCapture(0)
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Starting live audio and video processing. Press 'q' to quit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Display the video frame
        cv2.imshow('Live Video', frame)

        # Analyze facial expressions
        emotion = analyze_face_expression(frame)
        print("Facial Expression:", emotion)

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

    # Generate final report
    generate_report()

# Function to generate a final report
def generate_report():
    confidence_score = calculate_confidence()

    print("\n=== Final Interview Analysis Report ===")
    print(f"Total Filler Words Used: {total_filler_words}")
    print(f"Total Pauses Detected: {total_pauses}")
    print(f"Overall Sentiment Score (Avg): {np.mean(sentiment_scores) if sentiment_scores else 'N/A'}")
    print(f"Facial Expression Summary: {emotion_scores}")
    print(f"Confidence Level: {confidence_score} / 5")
    print("=======================================\n")

# Example Usage
if __name__ == "__main__":
    process_live_audio_video()
