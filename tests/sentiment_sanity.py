from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("models/roberta_with_features")
model.eval()

def predict(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=160)
    with torch.no_grad():
        out = model(**enc)
    probs = out.logits.softmax(-1).squeeze().tolist()
    return probs  # [p(neu), p(bad), p(good), p(irrelevant)]

print(predict("[OLD: Chiefs] [NEW: DOLPHINS]. Dolphins got fleeced."))
print(predict("[OLD: Chiefs] [NEW: DOLPHINS] Dolphins are so exciting"))
