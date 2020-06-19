import os
import argparse
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from models.encdec import Encoder, Decoder
from utils.iterator import trainIters, devIters
from utils.functions import showTrainingLoss
from data.loader import Preprocessor, ChildDataset, make_loader


START_token = "<s>"
END_token = "</s>"
START_int = 0 
END_int = 1

def run(train_feadir, train_textdir,
        dev_feadir, dev_textdir,
        lexicon_path, 
        fea_dim, num_classes, 
        embed_flag, embed_size,
        preprocess_input_flag, preprocess_mlp_dim,
        rnn_unit, hidden_size,
        encoder_layer, decoder_layer,
        transducer_flag, max_token_len,
        attention_mode, decode_mode,
        teacher_forcing_ratio,
        start_and_end,
        seed,
        batch_size,
        learning_rate,
        dropout_rate,
        epochs,
        save_path,
        checkpoint):
    
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    random.seed(seed)
    # device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")
        torch.manual_seed(seed)
    with open(lexicon_path, "rb") as f:
        lexicon = pickle.load(f)
    preproc = Preprocessor(fea_dim, start_and_end)
    dataset = ChildDataset(train_feadir, train_textdir, lexicon, preproc)
    train_loader = make_loader(dataset, batch_size)
    dev_set = ChildDataset(dev_feadir, dev_textdir, lexicon, preproc)
    dev_loader = make_loader(dev_set, batch_size)
    #---------------------Initialize parameters------------------------#
    # device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    input_size = fea_dim
    output_size = num_classes+2 # SOS+EOS+num_classes
    encoder = Encoder(input_size, hidden_size, encoder_layer, rnn_unit, 
        dropout_rate, device=device).to(device)
    decoder = Decoder(output_size, hidden_size, 2, hidden_size, 
        decoder_layer, max_token_len, transducer_flag, preprocess_input_flag, 
        preprocess_mlp_dim, embed_flag, embed_size, rnn_unit,
        attention_mode, decode_mode, dropout_rate, device=device).to(device)
    # 
    optimizer = torch.optim.Adam([{'params':encoder.parameters()}, 
      {'params':decoder.parameters()}], lr=learning_rate)
    # 
    criterion = nn.NLLLoss()

    epochs_count = []; train_losses = []; valid_losses = []
    train_scores = []; valid_scores = []
    best_score = 0.7
    best_loss = 3
    start_epoch = 1
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        best_loss = checkpoint["best_loss"]

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer"])
        # decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
        train_scores = checkpoint["train_scores"]
        valid_scores = checkpoint["valid_scores"]

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)
        print("********* Training epoch {}:".format(epoch))      
        train_loss, train_per = trainIters(train_loader, preproc, encoder, decoder, 
            optimizer, criterion, teacher_forcing_ratio)
        train_scores.append(train_per)
        train_losses.append(train_loss)
        print("-> Training loss = {:.4f}, Phone Error Rate = {:.4f} \n".format(train_loss, train_per))
        # 
        print("********* Validation for epoch {} on dev data:".format(epoch))
        valid_loss, valid_per = devIters(dev_loader, preproc, encoder, decoder,
            criterion, teacher_forcing_ratio)
        valid_scores.append(valid_per)
        valid_losses.append(valid_loss)
        print("-> Validation loss = {:.4f}, Phone Error Rate = {:.4f} \n".format(valid_loss, valid_per))
        if valid_per > best_score or valid_loss > best_loss:
            patience_counter += 1
        else:
            best_score = valid_per
            best_loss = valid_loss
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "encoder":encoder.state_dict(),
                        "decoder":decoder.state_dict(),
                        "optimizer":optimizer.state_dict(),
                        "best_score":best_score,
                        "best_loss":best_loss,
                        "epochs_count":epochs_count,
                        "train_losses":train_losses,
                        "valid_losses":valid_losses,
                        "train_scores":train_scores,
                        "valid_scores":valid_scores},
                        os.path.join(save_path, "best.pth.tar"))
        #   Save the model at each epoch
        torch.save({"epoch": epoch,
                    "encoder":encoder.state_dict(),
                    "decoder":decoder.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "best_score":best_score,
                    "best_loss":best_loss,
                    "epochs_count":epochs_count,
                    "train_losses":train_losses,
                    "valid_losses":valid_losses,
                    "train_scores":train_scores,
                    "valid_scores":valid_scores},
                    os.path.join(save_path, "model_{}.pth.tar".format(epoch)))
# 
    checkpoint = torch.load(os.path.join(save_path, "model_40.pth.tar"))
    showTrainingLoss(checkpoint, save_path)


#     f, ax = plt.subplots(ncols=2, nrows=1)
#     ax[0].plot(epochs_count, train_losses, "-r")
#     ax[0].plot(epochs_count, valid_losses, "-b")
#     ax[0].set_xlabel("epoch")
#     ax[0].set_ylabel("Losses")
#     ax[0].set_title("Losses")
# # 
#     ax[1].plot(epochs_count, train_scores, "-r")
#     ax[1].plot(epochs_count, valid_scores, "-b")
#     ax[1].set_xlabel("epoch")
#     ax[1].set_ylabel("PER")
#     ax[1].set_title("PER")
#     plt.savefig(os.path.join(save_path, 'training_loss'))
#     plt.show()

    # plt.figure()
    # plt.plot(epochs_count, train_losses, "-r")
    # plt.plot(epochs_count, valid_losses, "-b")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend(["Training loss",
    #             "Validation loss"])
    # plt.title("Cross entropy loss")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Train a phone alignment model")
    parser = argparse.ArgumentParser(description="Train the base model on phone_level")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    parser.add_argument("--train_feadir",
                        default="mannual_prepare/feats/fbank/train_man/fbank123.pickle")
    parser.add_argument("--train_textdir",
                        default="mannual_prepare/text/train_man/text.pickle")
    parser.add_argument("--dev_feadir",
                        default="mannual_prepare/feats/fbank/dev_man/fbank123.pickle")
    parser.add_argument("--dev_textdir",
                        default="mannual_prepare/text/dev_man/text.pickle")
    parser.add_argument("--lexicon_path", default="data/lexicon.pickle")
    parser.add_argument("--save_path", default="save/att_concat_1_nosil/")
    parser.add_argument("--dim_feas", default=123)
    parser.add_argument("--num_classes", default=33)
    parser.add_argument("--embed_flag", default=False)
    parser.add_argument("--embed_size", default=100)
    parser.add_argument("--preprocess_input_flag", default=False)
    parser.add_argument("--preprocess_mlp_dim", default=150)
    parser.add_argument("--rnn_unit",default='GRU')
    parser.add_argument("--hidden_size",default=100)
    parser.add_argument("--encoder_layer", default=2)
    parser.add_argument("--decoder_layer", default=2)
    parser.add_argument("--transducer_flag", default=False)
    parser.add_argument("--max_token_len", default=30)
    parser.add_argument("--attention_mode", default='concat')
    parser.add_argument("--decode_mode", default=1)
    parser.add_argument("--teacher_forcing_ratio", default=0.9)
    parser.add_argument("--start_and_end", default=True)  
    parser.add_argument("--seed", default=2020)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--learning_rate", default=0.001)
    parser.add_argument("--dropout_rate", default=0.0)
    parser.add_argument("--epochs", default=40)
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open(args.save_path+"model_config.json", 'w') as f:
        json.dump(args.__dict__, f, indent=1)

    run(os.path.normpath(args.train_feadir),
        os.path.normpath(args.train_textdir),
        os.path.normpath(args.dev_feadir),
        os.path.normpath(args.dev_textdir),
        os.path.normpath(args.lexicon_path),
        args.dim_feas,
        args.num_classes,
        args.embed_flag,
        args.embed_size,
        args.preprocess_input_flag,
        args.preprocess_mlp_dim,
        args.rnn_unit,
        args.hidden_size,
        args.encoder_layer,
        args.decoder_layer,
        args.transducer_flag,
        args.max_token_len,
        args.attention_mode,
        args.decode_mode,
        args.teacher_forcing_ratio,
        args.start_and_end,
        args.seed,
        args.batch_size,
        args.learning_rate,
        args.dropout_rate,
        args.epochs,
        os.path.normpath(args.save_path),
        args.checkpoint)

    

# os.system('python models/train.py --checkpoint "save/att_encdec_1/model_10.pth.tar"')
# os.system('python train_127_1.py')




