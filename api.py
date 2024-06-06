
import string

# import numpy as np
# import pandas as pd
import torch
# from torch import optim
# from torch.utils.data import Dataset, DataLoader
# import torch.nn.functional as F
# from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score
# import matplotlib.pyplot as plt
# import seaborn as sns
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import BadRequest

# Tentukan path tempat Anda menyimpan model dan tokenizer
model_save_path = "indobert_classification_model.pth"
tokenizer_save_path = "indobert_tokenizer"

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained(tokenizer_save_path)
class DocumentClassificationDataset():
    # Static constant variable
    LABEL2INDEX = {'pelecehan seksual': 0, 'kontak fisik langsung': 1, 'perilaku non verbal langsung': 2, 'cyber bullying': 3}
    INDEX2LABEL = {0: 'pelecehan seksual', 1: 'kontak fisik langsung', 2: 'Perilaku non verbal langsung', 3: 'cyber bullying'}
    NUM_LABELS = 4
# Initialize Model Configuration
config = {
    'num_labels': 4,  # Sesuaikan dengan jumlah label yang sesuai dengan model Anda
    # Jika ada konfigurasi lain yang perlu disesuaikan, tambahkan di sini
}

# Load Model
model = BertForSequenceClassification.from_pretrained('indobenchmark/indobert-base-p1', **config)

# Load State Dictionary
model.load_state_dict(torch.load(model_save_path, map_location=torch.device('cpu')))  # Sesuaikan dengan perangkat yang digunakan

# Set Model ke Mode Evaluasi
model.eval()

# Sekarang model siap digunakan untuk prediksi tanpa perlu pelatihan ulang


app=Flask(__name__)
CORS(app)
host_ip = '0.0.0.0'
def predict_text_bullying(model, tokenizer, text, max_seq_len=512, device='cpu'):
    model.eval()
    torch.set_grad_enabled(False)

    # Tokenize input text
    subwords = tokenizer.encode(text, add_special_tokens=True)[:max_seq_len]

    # Convert to tensor
    subword_tensor = torch.tensor(subwords, dtype=torch.long).unsqueeze(0).to(device)

    # Generate mask
    mask = torch.ones_like(subword_tensor).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(subword_tensor, attention_mask=mask)
        logits = outputs.logits

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    # Convert index to label
    label = DocumentClassificationDataset.INDEX2LABEL[predicted_label]

    return label,predicted_label


def preprocess_text(text):
    # Menghilangkan tanda baca
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    return text
# Endpoint untuk menerima teks dan memprediksi pembulian
@app.route('/predict_bullying', methods=['POST'])
def predict_bullying():
    try:
        # Dapatkan teks dari permintaan POST
        text = request.form.get('text', '')

        # Periksa apakah teks tidak kosong
        if not text:
            raise BadRequest("Teks tidak boleh kosong.")
        
        # Preprocess teks
        text = preprocess_text(text)

        # Prediksi pembulian
        predicted_bullying,id = predict_text_bullying(model, tokenizer, text, device='cpu')

        if id == 0:
            id = 7
        elif id == 2:
            id = 4
        elif id == 3:
            id = 6

        # Kirim kembali hasil prediksi sebagai respons JSON
        return jsonify({'predicted_bullying': predicted_bullying, "id":id}),200

    except BadRequest as e:
        # Tangani kesalahan jika teks tidak ada atau tidak valid
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
   app.run(host=host_ip)
