import json, joblib, pandas as pd, os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
app=FastAPI()
model=joblib.load("model.pkl")
df=pd.read_csv("real_estate.csv")
with open("chunks.json", encoding='utf-8') as f: chunks=json.load(f)

grok_key=os.getenv("GROK_API_KEY")
client=OpenAI(api_key=grok_key, base_url="https://api.x.ai/v1") if grok_key and grok_key.startswith("xai-") else None

class P(BaseModel):
    Location:str; Bedrooms:int; Bathrooms:int; Area_sqft:int; Age_years:int; Furnished:int
class R(BaseModel):
    budget:int
class C(BaseModel):
    question:str

def search(q):
    words=q.lower().split()
    s=sorted(chunks, key=lambda c: sum(w in c["text"].lower() for w in words), reverse=True)
    return [x["text"] for x in s[:3]]

@app.post("/predict")
def predict(p:P):
    price=model.predict(pd.DataFrame([p.dict()]))[0]
    return {"price":f"{price/1e6:.1f} Million PKR"}

@app.post("/recommend")
def recommend(r:R):
    filt=df[df.Price_PKR<=r.budget*1.2].sort_values("Price_PKR").head(5)
    return {"data": filt.to_dict(orient="records")}

@app.post("/chat")
def chat(c:C):
    ctx="\n".join(search(c.question))
    if client:
        try:
            ans=client.chat.completions.create(model="grok-3-mini", messages=[{"role":"system","content":f"Real estate expert. Context:{ctx}"},{"role":"user","content":c.question}]).choices[0].message.content
        except:
            ans=f"From knowledge base:\n{ctx}"
    else:
        ans=f"From knowledge base:\n{ctx}"
    return {"answer":ans}

app.mount("/", StaticFiles(directory="static", html=True), name="static")