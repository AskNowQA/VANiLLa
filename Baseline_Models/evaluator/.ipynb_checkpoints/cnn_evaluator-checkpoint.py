from train_eval.cnn_train_eval import *
from data.dataset import *
from models.cnn_model import *
#from train_eval.rnn_train_eval import *
#from models.rnn_model import *
#from train_eval.rnn_attn_train_eval import *
#from models.rnn_attn_model import *
from utils.cnn_blue import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data, test_data, val_data, QnA_vocab, Ans_Sen_vocab = loadDataset()

INPUT_DIM = len(QnA_vocab)
OUTPUT_DIM = len(Ans_Sen_vocab)
EMB_DIM = 100
HID_DIM = 512 # each conv. layer has 2 * hid_dim filters
ENC_LAYERS = 10 # number of conv. blocks in encoder
DEC_LAYERS = 10 # number of conv. blocks in decoder
ENC_KERNEL_SIZE = 3 # must be odd!
DEC_KERNEL_SIZE = 3 # can be even or odd
ENC_DROPOUT = 0.25
DEC_DROPOUT = 0.25
TRG_PAD_IDX = Ans_Sen_vocab.stoi[Ans_Sen.pad_token]
BATCH_SIZE = 100

print("==> Building Encoder")

enc = Encoder(INPUT_DIM, EMB_DIM, HID_DIM, ENC_LAYERS, ENC_KERNEL_SIZE, ENC_DROPOUT, device, QnA_vocab.vectors)

print("==> Building Decoder")

dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM, DEC_LAYERS, DEC_KERNEL_SIZE, DEC_DROPOUT, TRG_PAD_IDX, device, Ans_Sen_vocab.vectors)

print("==> Building Seq2Seq Model")

model = Seq2Seq(enc, dec, device).to(device)

model.load_state_dict(torch.load('checkpoints/cnn-model.pt'))

bleu_score = calculate_bleu(test_data, QnA, Ans_Sen, model, device)

print('BLEU score = {:.2f}'.format(bleu_score*100))