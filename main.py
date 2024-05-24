from typing import Dict
from fastapi import FastAPI, HTTPException
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import configparser
import json
import joblib
from keras.models import load_model
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

# TONE CLASSIFIER
classifierTone = joblib.load("tone_classifier_model.pkl")
vectorizerTone = joblib.load("tfidf_vectorizer.pkl")

# MLP CLASSIFIER
MLPModel = load_model("ModeloMLP.keras", compile=False)

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the file
config.read('CONFIG.cfg')

# Access the dictionary
label_urgency_dict = {}
for key in config['label_urgency_dict']:
    label_urgency_dict[key] = int(config['label_urgency_dict'][key])
@dataclass
class Email():
     Subject: str
     Body: str

# Example ConfigType model for validation
@dataclass
class ConfigType():
    config: Dict[str, int]

@dataclass
class Urgency():
     urgencyRating : int
     emailTopics : list[str]
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def RateEmailTone(emailInput: Email) -> int:
     # Email to predict
     email = """
          Subject:
          {Subject}
          Body:
          {Body}""".format(Subject=emailInput.Subject, Body=emailInput.Body)
     # Vectorize the email
     email_vectorized = vectorizerTone.transform([email])

     # Predict the tone of the email
     predicted_tone = classifierTone.predict(email_vectorized)

     toneClass = 1 if predicted_tone[0] == "formal" else - \
     1 if predicted_tone[0] == "casual" else 0
     return toneClass

def RateEmailTopic(email: Email) -> Urgency:
     sequence_to_classify = """
          Subject
          {Subject}
          Body
          {Body}""".format(Subject=email.Subject, Body=email.Body)

     classDict = label_urgency_dict(sequence_to_classify, list(
     label_urgency_dict.keys()), multi_label=True)
     filtered_predictions = [(label, score) for label, score in zip(
     classDict['labels'], classDict['scores']) if score > 0.3]

     sorted_predictions = sorted(
     filtered_predictions, key=lambda x: x[1], reverse=True)

     top_3_classifications = [label for label, _ in sorted_predictions[:3]]

     urgency_ratings = [label_urgency_dict[classification]
                    for classification in top_3_classifications]
     if not top_3_classifications:
       top_3_classifications = ["other"]
       urgency_rating = 0
     else:
       urgency_rating = max(urgency_ratings)
     print("Still missing")
     return {Urgency(urgency_rating,top_3_classifications)}

@app.get("/")
async def SendJSONTemplate():
     return Email("Your Subject", "Your Body")


@app.get("/config")
async def SendJSONTemplate():
     return label_urgency_dict
# Example endpoint to receive the config data
@app.post("/config")
async def send_json_template(config_data: ConfigType):
    # Assuming config_data is a dictionary containing the entire config data
    try:
        # Generate the configuration string in the specified format
        config_string = "[label_urgency_dict]\n"
        for key, value in config_data.config.items():
            config_string += f"{key} = {value}\n"

        # Write the generated configuration string to the CONFIG.cfg file
        with open("CONFIG.cfg", "w") as file:
            file.write(config_string)

        global label_urgency_dict
        label_urgency_dict = config_data.config
        return {"message": "Config updated successfully"}  # Return success message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Return error message with status code 500 for internal server error
   
@app.post("/")
async def RateEmail(email:Email):
          TopicUrgency = RateEmailTopic(email)
          if (TopicUrgency == 5) or (TopicUrgency == 0):
               return TopicUrgency
          TopicUrgency.urgencyRating = TopicUrgency.urgencyRating/5
          ToneUrgency = RateEmailTone(email)/5
          TopicUrgency.urgencyRating = MLPModel.predict([TopicUrgency.urgencyRating,ToneUrgency])
          return {TopicUrgency}