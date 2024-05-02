from typing import Dict
from fastapi import FastAPI, HTTPException
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import configparser
import json
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

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
        
        return {"message": "Config updated successfully"}  # Return success message
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))  # Return error message with status code 500 for internal server error
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

     urgency_ratings = [label_urgency_dict[classification] for classification in top_3_classifications]
     if not top_3_classifications:
        top_3_classifications = ["other"]
        urgency_ratings=0
     urgency_rating = max(urgency_ratings)
     return {Urgency(urgency_rating,top_3_classifications)}