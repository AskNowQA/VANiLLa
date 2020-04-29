from train_eval.rnn_train_eval import *
from data.dataset import *
from models.rnn_model import *
from utils.rnn_blue import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data, val_data, QnA_vocab, Ans_Sen_vocab = loadDataset()

INPUT_DIM = len(QnA_vocab)
OUTPUT_DIM = len(Ans_Sen_vocab)
ENC_EMB_DIM = 100
DEC_EMB_DIM = 256
HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TRG_PAD_IDX = Ans_Sen_vocab.stoi[Ans_Sen.pad_token]

print("==> Building Encoder")

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, ENC_DROPOUT, device, QnA_vocab.vectors)

print("==> Building Decoder")

dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, DEC_DROPOUT, device)

print("==> Building Seq2Seq Model")

model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('checkpoints/rnn-model.pt'))

bleu_score = calculate_bleu(test_data, QnA, Ans_Sen, model, device)

print('BLEU score = {:.2f}'.format(bleu_score*100))