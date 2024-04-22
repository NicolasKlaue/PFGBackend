from fastapi import FastAPI
from dataclasses import dataclass
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


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
     sequence_to_classify = """
     Subject
     {Subject}
     Body
     {Body}""".format(Subject = email.Subject, Body = email.Body)
     candidateLabels = ['energy infrastructure',
                        'other',
                        'government & politics',
                        'legal affairs',
                        'calendar & scheduling',
                        'employment',
                        'personal & social',
                        'customer support',
                        'business document',
                        'energy trading',
                        'project management',
                        'meetings & events',
                        'project-specific',
                        'human resources',
                        'contract management',
                        'marketing & promotion',
                        'information technology',
                        'external affairs',
                        'media & press' 'finance',
                        'corporate governance',
                        'energy services',
                        'organization',
                        'utilities',
                        'phone communication',
                        'safety & emergency',
                        'market research',
                        'email management',
                        'education & training']
     classDict = classifier(sequence_to_classify, candidateLabels,multi_label=True)
     urgency= Urgency(int(candidateLabels.index(classDict['labels'][0]) + 1 ),classDict['labels'][0])
     return urgency