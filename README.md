# Child Speech Sound Error Detection
Please try eval.py
The checkpoint "checkpoint-5500" is too big to upload. Please contact me to share it via other methods such as google drive.
This checkpoint leverages the pretrained checkpoint wav2vec2-large-lv60 and is fine-tuned on CommonVoice and CUCHILD to recognize phonetic labels in Cantonese. When using the model make sure that your speech input is sampled at 16kHz. Note that the model outputs a string of phonetic labels.


