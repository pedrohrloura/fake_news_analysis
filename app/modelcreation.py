import numpy as np
import pandas as pd
import pickle
import torch
import transformers as ppb  # pytorch transformers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

# Configurar o logger para exibir informações de progresso
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Carregar os dados de treinamento do arquivo CSV
train_filename = './welfake_dataset/WELFake_Dataset.csv'
train_news = pd.read_csv(train_filename)

# Verificar se CUDA está disponível e definir o dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Carregar o modelo pré-treinado BERT
model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
model.to(device)  # Transferir o modelo para a GPU se disponível

# Definir o tamanho máximo da sequência
max_seq_length = 128

# Função para pré-processar os dados de texto usando BERT
def preprocess_with_bert(text):
    tokenized = tokenizer.encode(text, add_special_tokens=True)
    if len(tokenized) > max_seq_length:
        tokenized = tokenized[:max_seq_length]
    padded = tokenized + [0] * (max_seq_length - len(tokenized))
    input_ids = torch.tensor([padded]).to(device)
    return input_ids

train_news = train_news.dropna(subset=['Statement']).reset_index(drop=True)
train_news['Statement'] = train_news['Statement'].apply(lambda x: x[:max_seq_length] if len(x) > max_seq_length else x)

# Aplicar a função de pré-processamento aos dados de treinamento
train_inputs = train_news['Statement'].apply(preprocess_with_bert)

train_embeddings = []
for i, input_ids in enumerate(train_inputs):
    logger.info(f"Processing training example {i+1}/{len(train_inputs)}")
    with torch.no_grad():
        outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
    train_embeddings.append(last_hidden_states.cpu().numpy())

# Converter embeddings para numpy arrays e achatar para 2Df
X_train = np.vstack(train_embeddings).reshape(len(train_embeddings), -1)

# Preparar os rótulos
y_train = train_news['Label']

X_train = X_train[:1000]
y_train = y_train[:1000]

# Treinar um classificador nos embeddings BERT
logger.info("Training classifier...")
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
logger.info("Classifier training completed.")

# Salvar o modelo treinado
model_file = 'WELFake_model.sav'
pickle.dump(classifier, open(model_file, 'wb'))
logger.info(f"Model saved as {model_file}")