from fastapi import FastAPI
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

import configparser

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

@app.get("/")
async def SendJSONTemplate():
     return Email("Your Subject", "Your Body")

@app.post("/")
async def RateEmail(email:Email):
     sequence_to_classify = """
     Subject
     {Subject}
     Body
     {Body}""".format(Subject = email.Subject, Body = email.Body)

     classDict = classifier(sequence_to_classify, list(label_urgency_dict.keys()), multi_label=True)
     print(classDict)
     filtered_predictions = [(label, score) for label, score in zip(classDict['labels'], classDict['scores']) if score > 0.3]

     sorted_predictions = sorted(filtered_predictions, key=lambda x: x[1], reverse=True)

     top_3_classifications = [label for label, _ in sorted_predictions[:3]]

     if not top_3_classifications:
        top_3_classifications = ["other"]
     urgency_ratings = [label_urgency_dict[classification] for classification in top_3_classifications]
     urgency_rating = max(urgency_ratings)
     return {"urgencyRating": urgency_rating, "emailTopics": top_3_classifications}