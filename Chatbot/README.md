# EmoLingo: An Emotion-Aware Conversational Agent for Personalized Emotional Support

---

This repository demonstrates a Speech Emotion Recognition (SER) pipeline integrated into a conversational AI system. The system takes in a speech input, transcribes it to text, detects emotional cues in the speech signal, then uses GPT to generate an appropriate response based on both the context and the detected emotion, and finally converts the generated reply back into speech.

--- 

## Installation and Execution Guide

1. **Clone or Download** this repository
2. **Mount Google Drive**
    ```
    from google.colab import drive
    drive.mount('/content/drive')
    ```
3. **Install the necessary Libraries**
    ```
    ! pip install transformers torch speechrecognition
    ! pip install TTS torch
    ! pip install numpy==1.23.5
    ```
    ```
    ! pip install cutlet
    ```
    ```
    %pip install openai
    ```
    ```
    ! pip install git+https://github.com/openai/whisper.git
    ! sudo apt update && sudo apt install ffmpeg
    ```
    ```
    !pip -q install pydub
    !apt-get install -y ffmpeg
    ```
4. **Import** the required libraries 
5. **Run the following cell** according to the Program Flow below
6. **Load the Pre-trained Model** for each corresponding section
    ```
    json_path = "/content/drive/MyDrive/ML Team 35/Codes/Chatbot/CNN_model.json" 
    weights_path = "/content/drive/MyDrive/ML Team 35/Codes/Chatbot/CNN_model_weights.h5" 
    scaler_path = '/content/drive/MyDrive/ML Team 35/Codes/Chatbot/scaler.pickle'
    encoder_path = '/content/drive/MyDrive/ML Team 35/Codes/Chatbot/encoder.pickle'
    ```
    Make sure the **folder path** of the pre-trained models are correct, and that the **root folder** (in this case *ML Team 35*) must be located in the main page of *MyDrive* in your google drive
---

## Program Flow

1. [Importing Libraries and Models](#importing-libraries-and-models)
2. [Speech Emotion Recognition (SER)](#speech-emotion-recognition-ser)
3. [Language Detection](#language-detection)
4. [Speech-To-Text (STT)](#speech-to-text-stt)
5. [GPT 3.5](#data-preparation)
6. [Text-To-Speech (TTS)](#text-to-speech-tts)
7. [Chatbot](#chatbot)

---

## Importing Libraries and Models

We begin by importing the general-purpose libraries, such as:
    
    numpy, torch, time, os, shutil, IPython.display, tqdm, time, google.colab, base64, io, pydub

We also need to import some libraries used for our chatbot pipeline's features:
  - **STT**: `speech_recognition`
  - **Language Detection**: `whisper`
  - **SER**: `tensorflow.keras`, `pickle`, `librosa`
  - **GPT 3.5**: `openai`
  - **TTS**: `TTS.api`

--- 

## Speech Emotion Recognition (SER)

  1. Load the pre-trained model, parameters, scaler, and encoder
  2. Record the Audio 
  3. Extract features from the audio (ZCR, RMSE, MFCC) with the help of *librosa* library
  4. Preprocess the recorded audio with the pre-loaded scaler and encoder
  5. Predict the emotion based on the preprocessed audio as well as the extracted features

---

## Language Detection

Load the model from the *whisper* library

---

## Speech-To-Text (STT)

  1. Load the *Recognizer* model from the *speech_recognition* library
  2. Convert the recorded audio into text

---

## GPT 3.5

  1. Initialize an instance of the GPT 3.5 API with the appropriate API key
  2. Initialize the predefined instructions for language and emotion
     - **language**: sets predefined instructions for the assistant based on the specified language, set English by default if the language is not listed
     - **emotion**: adjusts the system instructions to reflect the emotion provided to respond with empathy based on emotional cues
  3. Handling unintelligible audio responses with a predefined response if the audio input fails to process
  4. Combine the **message**, **language**, and **emotion** to send to the API where it uses GPT 3.5's *streaming mode* to generate responses dynamically in chunks
  5. If the user's audio was unintelligible, it returns a predefined apology message, otherwise it streams and return a dynamically generated response from GPT 3.5 based on the message, language, and emotion context

---

## Text-To-Speech (TTS)

  1. Load the Text-To-Speech model with the help of the TTS library
  2. Checks whether the specified language is supported by the system or not, if it isn't then it will be set to *English* by default and set the emotion to *Neutral*
  3. Set the output directories and file paths
  4. Generates audio with given **text**, **language**, **emotion**, and **voice sample** with the *tts.tts_to_file()* function
  5. Save the audio locally in Colab and create a copy to Google Drive for backup

---

## Chatbot

This chatbot implements a complete voice-based chatbot pipeline that combines speech-to-text (STT), emotion recognition (SER), GPT-generated replies, and text-to-speech (TTS) to deliver an interactive conversational experience. It processes audio input from the user, detects the language and emotion, generates an empathic response using GPT-3.5, converts the reply into speech, and plays the audio response back to the user.

### Steps in the Process
  
  1. Recorrds audio for a specified duration and saves the audio in Colab and Google Drive
  2. Transcribe the audio and detect the language with the pre-loaded *whisper* model used for language detection
  3. Converts the recorded audio into text in the detected language with the STT model
  4. Analyzes the audio to identify the emotion with the SER model
  5. Generate GPT reply 
     - Send the transcribed text and detected emotion to the GPT 3.5 API which includes the prompt *reply in one sentence with more concerning with speaker's emotion*
     - Customizes the assistant's tone based on the detected emotion
     - Receives the text reply from GPT
  6. Converts the GPT-generated reply text into speech using the TTS API and a pre-specified speaker voice sample *LiShen.wav* for voice cloning, and saves the generated audio both in Colab and Google Drive
  7. Plays the generated speech response back to the user using *IPython audio* library

---