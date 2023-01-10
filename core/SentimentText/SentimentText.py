import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentText:
    def init(self, model_name_or_path = "distilbert-base-uncased-finetuned-sst-2-english", device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.device = "cuda" if torch.cuda.is_available() and device == 'cuda' else "cpu"

    def analysis(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        return self.model.config.id2label[predicted_class_id]
    