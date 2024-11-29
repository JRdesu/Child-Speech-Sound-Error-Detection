from flask import Flask, render_template, request, jsonify, redirect, url_for
from typing import List, Union
import os
import uuid
import torch
import librosa
from kaldialign import align
from lexicon_phn_ipa import lexicon

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

app = Flask(__name__)

# Set the base upload folder
app.config['BASE_UPLOAD_FOLDER'] = 'uploads'  # Change this path as needed

# Create the upload directory if it doesn't exist
os.makedirs(app.config['BASE_UPLOAD_FOLDER'], exist_ok=True)

# Define the target words
target_words = ["baak_tin_ngo", "baak_tou", "ciu_kap_si_coeng", "coi_hung"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============== Load trained model and processor ==================
# Initialize model and processor
model_checkpoint = "./checkpoints/checkpoint-6000"
model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint)
processor = Wav2Vec2Processor.from_pretrained("./checkpoints/checkpoint-6000")

tokenizer = processor.tokenizer
model.to(device)
model.eval()

#=================== Pad token ======================
PAD_ID = tokenizer.encode("<pad>")[0]

def collapse_tokens(tokens: List[Union[str, int]]) -> List[Union[str, int]]:
    prev_token = None
    out = []
    for token in tokens:
        if token != prev_token and prev_token is not None:
            out.append(prev_token)
        prev_token = token
    return out


def clean_token_ids(token_ids: List[int]) -> List[int]:
    """
    Remove [PAD] and collapse duplicated token_ids
    """
    token_ids = [x for x in token_ids if x not in [PAD_ID]]
    token_ids = collapse_tokens(token_ids)
    return token_ids

def evaluate(filename):
    # file = "/home/jiarui/Documents/asr/cuchild/atypical/CC___2017052411/kingjyu_kh_k_deaspiration.wav"
    # file = "/home/jiarui/Documents/asr/cuchild/demo/gamjyu_k_t_fronting+m_ng_backing.wav"
    speech, sr = librosa.load(filename, sr=16000)
    speech = librosa.util.fix_length(speech, size=len(speech)+1600, mode='edge')
    input_values = processor(speech, sampling_rate=16000).input_values[0]
    inputs = torch.tensor(input_values).unsqueeze(0)
# 
    with torch.no_grad():
        logits = model(inputs.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_ids1 = clean_token_ids(predicted_ids[0].int().tolist())
        predicted_chr1 = tokenizer.decode(predicted_ids1, group_tokens=False)
        return predicted_chr1
    

def filter_consonant(text):
    consonants = ['l', 'k', 'ŋ', 'j', 'tʃ', 'tʃʰ', 'f', 'p', 'n', 'pʰ', 'h', 'w', 'm', 'ʃ', 't', 'kʰ', 'tʰ', 's']
    coda = ['k', 't', 'p', 'n', 'm', 'ŋ']
    pos = []
    tokens = []
    phns = text.split(' ')
    num_phns = len(phns)
    if num_phns == 1 and text in consonants:
        pos = [0]
        return pos, text
    for i, t in enumerate(phns):
        if t in consonants:
            if i == 0 and t in consonants:
                pos.append(0)
                tokens.append(t)
            elif i == 1 and t == 'w' and tokens[0] in ['k', 'kʰ']:
                tokens[0] = tokens[0]+'w'  
            elif t in coda and i < num_phns-1 and phns[i+1] in consonants:
                continue
            elif t in coda and i == num_phns-1 and (phns[i-1] not in consonants):
                continue
            else:
                pos.append(i)
                tokens.append(t)
    return pos, tokens

def total_err(alis):
    counts = {}
    corrects = {}
    totals = {}
    for ali in alis:
        target, real = ali
        if target not in counts:
            counts[target] = 1
            corrects[target] = 0
        else:
            counts[target] += 1
        if target == real:
            corrects[target] += 1
    for k, v in counts.items():
        totals[k] = f"{corrects[k]}/{counts[k]}"
    return totals

#===============================================================

@app.route('/')

@app.route('/upload', methods=['GET', 'POST'])
def upload_audio():
    if request.method == 'POST':
        files = request.files.getlist('audio_files')  # Get the list of uploaded files

        # Check if the correct number of files are uploaded
        if len(files) != len(target_words):
            return jsonify({'error': 'Number of files must match the number of target words'}), 400

        alis = {}
        sum_alis = []
        for i, file in enumerate(files):
            audio_id = str(uuid.uuid4())  # Generate a unique identifier for each file
            filename = f"{target_words[i]}_{audio_id}{os.path.splitext(file.filename)[1]}"
            file_path = os.path.join(app.config['BASE_UPLOAD_FOLDER'], filename)
            file.save(file_path)  # Save the file

            reference_text = ' '.join(lexicon[target_words[i]])
            decoded_text = evaluate(file_path)
            ref_pos, ref_consonants = filter_consonant(reference_text)
            dec_pos, dec_consonants = filter_consonant(decoded_text)
            alis[target_words[i]] = align(ref_consonants, dec_consonants, '*')
            sum_alis.extend(align(ref_consonants, dec_consonants, '*'))

        totals = total_err(sum_alis)
        # Store transcript in session or database for retrieval in results page
        # For simplicity, we'll use a global variable here (not recommended for production)
        global results, sum_results
        results = alis
        sum_results = totals
        return redirect(url_for('show_results'))
    
    return render_template('upload.html', target_words=target_words)

@app.route('/results', methods=['GET'])
def show_results():
    global results
    return render_template('results.html', results=results, sum_results=sum_results)

if __name__ == "__main__":
    app.run(debug=True)




