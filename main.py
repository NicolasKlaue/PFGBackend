from fastapi import FastAPI
from dataclasses import dataclass

@dataclass
class Email():
     Subject: str
     Body : str

app = FastAPI()

@app.post("/")
async def RateEmail(email:Email) -> Email:
     return email