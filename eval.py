from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2Config

import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import math

model_checkpoint = "./checkpoint/checkpoint-5500"
model = Wav2Vec2ForCTC.from_pretrained(model_checkpoint)
# Load the configuration of a pre-trained model
config = Wav2Vec2Config.from_pretrained(model_checkpoint)
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
# Wav2Vec2Processor.save_pretrained(processor, model_checkpoint)
processor = Wav2Vec2Processor.from_pretrained("./checkpoint/checkpoint-5500")
tokenizer = processor.tokenizer

model.to("cuda")
model.eval()

PAD_ID = tokenizer.encode("<pad>")[0]
# EMPTY_ID = tokenizer.encode(" ")[0]

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


#=======================================================================
import librosa
speech, sr = librosa.load('demo/tai_zi_zap.wav')
input_values = processor(speech, sampling_rate=16000).input_values[0]
inputs = torch.tensor(input_values).unsqueeze(0)

with torch.no_grad():
    logits = model(inputs.to("cuda")).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_ids1 = clean_token_ids(predicted_ids[0].int().tolist())
    predicted_chr1 = tokenizer.decode(predicted_ids1, group_tokens=False)
    print(f"{predicted_chr1}\n")
    # the output should be "tʰ aː i ʃ iː tʃ ɐ p"

  
