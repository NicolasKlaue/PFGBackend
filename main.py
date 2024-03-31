from fastapi import FastAPI
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

@dataclass
class Email():
     Subject: str
     Body: str

@dataclass
class Urgency():
     urgencyRating : int
     emailTopics : str
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/")
async def RateEmail(email:Email) -> Urgency:
     urgency= Urgency(3,"informative")
     return urgency