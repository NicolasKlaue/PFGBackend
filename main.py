from fastapi import FastAPI
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

@dataclass
class Email():
     Subject: str
     Body : str

app = FastAPI()

@app.post("/")
async def RateEmail(email:Email) -> Email:
     return email