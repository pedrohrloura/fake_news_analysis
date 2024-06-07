import pickle
import torch
import transformers as ppb

# Carregar o modelo treinado
model_file = 'WELFake_model.sav'
classifier = pickle.load(open(model_file, 'rb'))

tokenizer = ppb.BertTokenizer.from_pretrained('bert-base-uncased')

model = ppb.BertModel.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transferir o modelo para o dispositivo
model.to(device)

max_seq_length = 128

# Função para pré-processar os dados de entrada para previsão
def preprocess_with_bert_for_prediction(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    if len(tokenized) > max_seq_length:
        tokenized = tokenized[:max_seq_length]
    padded = tokenized + [0] * (max_seq_length - len(tokenized))
    input_ids = torch.tensor([padded]).to(device)
    return input_ids

# Função para fazer previsões com o modelo BERT
def predict_with_bert(text):
    input_ids = preprocess_with_bert_for_prediction(text)
    with torch.no_grad():
        outputs = model(input_ids.to(device))
    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states.cpu().numpy().squeeze()
    prediction = classifier.predict(embeddings.reshape(1, -1))
    probabilities = classifier.predict_proba(embeddings.reshape(1, -1))
    predicted_label = prediction[0]
    confidence = probabilities.max() * 100
    if predicted_label == 0:
        predicted_text = "Fake News"
    elif predicted_label == 1:
        predicted_text = "Notícia verdadeira"

    print(f"Previsão: {predicted_text}, Confiança: {confidence:.2f}%")
    return predicted_label

while True:
    user_input = input("Digite um texto para fazer uma previsão (ou 'sair' para encerrar): ")
    if user_input.lower() == 'sair':
        print("Encerrando o programa...")
        break
    prediction = predict_with_bert(user_input)