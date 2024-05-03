from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

app = FastAPI()

# Carga del modelo y tokenizer
model_path = "models"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

class TextData(BaseModel):
    text: str

def prepare_text_for_inference(text: str, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding['input_ids'], encoding['attention_mask']

def make_prediction(text: str, model, tokenizer, device):
    model.eval()

    input_ids, attention_mask = prepare_text_for_inference(text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

    return probabilities.cpu().numpy(), predicted_class.cpu().numpy()

@app.post("/predict/")
async def predict(data: TextData):
    probabilities, predicted_class = make_prediction(data.text, model, tokenizer, device)
    return {
        "probabilities": probabilities.tolist(),
        "predicted_class": int(predicted_class)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
