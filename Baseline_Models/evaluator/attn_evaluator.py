from train_eval.attn_train_eval import *
from data.dataset import *
from models.attn_model import *
from utils.blue_scorer import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data, val_data, QnA_vocab, Ans_Sen_vocab = loadDataset()

INPUT_DIM = len(QnA_vocab)
OUTPUT_DIM = len(Ans_Sen_vocab)
ENC_EMB_DIM = 300
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
TRG_PAD_IDX = Ans_Sen_vocab.stoi[Ans_Sen.pad_token]
SRC_PAD_IDX = QnA_vocab.stoi[QnA.pad_token]

print("==> Building Encoder")

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT, device, QnA_vocab.vectors)

print("==> Building Decoder")
attn = Attention(ENC_HID_DIM, DEC_HID_DIM, device)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, device, attn)

print("==> Building Seq2Seq Model")

model = Seq2Seq(enc, dec, SRC_PAD_IDX, device).to(device)

model.load_state_dict(torch.load('checkpoints/attn-model.pt'))

bleu_score = data_scorer(test_data, QnA, Ans_Sen, model, device)

print('BLEU score = {:.2f}'.format(bleu_score*100))