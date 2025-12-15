import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

base_classifier = pipeline(
    "text-classification",
    model="roberta-base-openai-detector",
    return_all_scores=True
)

best_model_path = './xlm_roberta_ai_detector/best_model'

trained_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
trained_tokenizer = AutoTokenizer.from_pretrained(best_model_path)

trained_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained_model.to(device)


def get_ai_probability(text, classification_model):
    if classification_model == '1':
        results = base_classifier(text)

        labls = ['Сгенерирован ИИ', 'Написан человеком']

        for i in range(len(results[0])):
            results[0][i]['label'] = labls[i]

    elif classification_model == '2':

        inputs = trained_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = trained_model(**inputs)
            logits = outputs.logits

        results = list(torch.softmax(logits, dim=-1)[0])
        print(results)
        results = [[{'label': 'Сгенерирован ИИ', 'score': float(results[0])}, {'label': 'Написан человеком', 'score': float(results[1])}]]

    probability_percent = results[0]

    return probability_percent
