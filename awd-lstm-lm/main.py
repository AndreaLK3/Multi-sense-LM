import argparse
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn

import data
import model
from asgd import ASGD
from model_save import model_load, model_save, model_state_save
from sys_config import BASE_DIR, CKPT_DIR, CACHE_DIR

from utils import batchify, get_batch, repackage_hidden
import os, sys

# allowing for use of tools in the parent folder
sys.path.append(os.path.join(os.getcwd(), '..', ''))
import Graph.DefineGraph as DG
import WordEmbeddings.ComputeEmbeddings as CE
import Filesystem as F
import pandas as pd


parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=800,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1882,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default='BLIBLU' + '.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str, default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
parser.add_argument("-g", "--gpu", required=False,
                    default='1', help="gpu on which this experiment runs")
parser.add_argument("-server", "--server", required=False,
                    default='ford', help="server on which this experiment runs")
parser.add_argument("-asgd", "--asgd", required=False,
                    default='True', help="server on which this experiment runs")
args = parser.parse_args()
args.tied = True

if args.server is 'ford':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print("\nThis experiment runs on gpu {}...\n".format(args.gpu))

###############################################################################
print("torch:", torch.__version__)
#if torch.__version__ != '0.1.12_2':
print("Cuda:", torch.version.cuda) # * modified: these statistics have been moved in PyTorch 1.5
# print("CuDNN:", torch.backends.cudnn.version()) # if it's not present it causes an error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: {}'.format(device))
###############################################################################
global model, criterion, optimizer

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    args.cuda = True
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
else:
    args.cuda = False
    print('No cuda! device is cpu :)')

###############################################################################
# Load data
###############################################################################
print('Base directory: {}'.format(BASE_DIR))
# fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
fn = 'corpus.{}'.format(args.data)
fn = fn.replace('data/', '').replace('wikitext-2', 'wt2')

fn_path = os.path.join(CACHE_DIR, fn)
if os.path.exists(fn_path):
    print('Loading cached dataset...')
    corpus = torch.load(fn_path)
else:
    print('Producing dataset...')
    datapath = os.path.join(BASE_DIR, args.data)
    corpus = data.Corpus(datapath)
    torch.save(corpus, fn_path)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

vocabulary = corpus.dictionary

###############################################################################
# Build the model
###############################################################################
from splitcross import SplitCrossEntropyLoss

criterion = None

ntokens = len(corpus.dictionary)
# added
os.chdir('..')
graph_dataobj = DG.get_graph_dataobject(new=False, method=CE.Method.FASTTEXT, slc_corpus=False).to(device)
globals_vocabulary_fpath = os.path.join(F.FOLDER_VOCABULARY, F.VOCABULARY_OF_GLOBALS_FILE)
vocab_h5 = pd.HDFStore(globals_vocabulary_fpath, mode='r')
globals_vocabulary_df = pd.read_hdf(globals_vocabulary_fpath, mode='r')
globals_vocabulary_wordList = globals_vocabulary_df['word'].to_list().copy()
os.chdir('awd-lstm-lm')
variant_flags_dict = {'include_globalnode_input':False, 'include_sensenode_input':False}
#
model = model.AWD(args.model, ntokens, args.emsize, args.nhid,
                       args.nlayers, graph_dataobj, variant_flags_dict, globals_vocabulary_wordList,
                       args.dropout, args.dropouth,
                       args.dropouti, args.dropoute, args.wdrop, args.tied)

###
if args.resume:
    print('Resuming model ...')
    model, criterion, optimizer, vocab, val_loss, config = model_load(args.resume)
    optimizer.param_groups[0]['lr'] = args.lr
    model.dropouti, model.dropouth, model.dropout, args.dropoute = args.dropouti, args.dropouth, args.dropout, args.dropoute
    if args.wdrop:
        from weight_drop import WeightDrop

        for rnn in model.rnns:
            if type(rnn) == WeightDrop: rnn.dropout = args.wdrop
            elif rnn.zoneout > 0: rnn.zoneout = args.wdrop
###
if not criterion:
    splits = []
    if ntokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif ntokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using splits {}'.format(splits))
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

# if torch.__version__ != '0.1.12_2':
#     print([(name, p.device) for name, p in model.named_parameters()])
###
if args.cuda:
    model = model.cuda()
    criterion = criterion.cuda()
###
params = list(model.parameters()) + list(criterion.parameters())
trainable_parameters = [p for p in model.parameters() if p.requires_grad]
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
print('Args:', args)
print('Model total parameters:', total_params)


###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train():
    # Turn on training mode which enables dropout.
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activation Regularization
        if args.alpha: loss = loss + sum(
            args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if args.beta: loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

        ####################################
        if args.cuda:
            try:
                torch.cuda.empty_cache()
                # print('torch cuda empty cache')
            except:
                pass
        ####################################


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000

print('Starting training......')
# At any point you can hit Ctrl + C to break out of training early.
try:
    optimizer = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wdecay)  # params not trainable params... (?)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    for epoch in range(1, args.epochs+1):
        print('Starting epoch {}'.format(epoch))
        epoch_start_time = time.time()
        ####################################
        # memory debug
        print('Memory before train')
        if args.cuda:
            print(torch.cuda.get_device_properties(device).total_memory)
            print(torch.cuda.memory_cached(device))
            print(torch.cuda.memory_allocated(device))
        ####################################
        train()
        ####################################
        print('Memory after train')
        if args.cuda:
            print(torch.cuda.get_device_properties(device).total_memory)
            print(torch.cuda.memory_cached(device))
            print(torch.cuda.memory_allocated(device))
        ####################################
        if args.cuda:
            try:
                torch.cuda.empty_cache()
                # print('torch cuda empty cache')
            except:
                pass
        ####################################
        if 't0' in optimizer.param_groups[0]:  # if ASGD
            tmp = {}
            for prm in model.parameters():
                if prm in optimizer.state.keys():
                    # tmp[prm] = prm.data.clone()
                    tmp[prm] = prm.data.detach()
                    # tmp[prm].copy_(prm.data)
                    # if 'ax' in optimizer.state[prm]:  # added this line because of error: File "main.py", line 268, in <module> prm.data = optimizer.state[prm]['ax'].clone() KeyError: 'ax'
                    # prm.data = optimizer.state[prm]['ax'].clone()
                    prm.data = optimizer.state[prm]['ax'].detach()

                # else:
                #     print(prm)

                    # prm.data = optimizer.state[prm]['ax'].clone()
                    # prm.data = optimizer.state[prm]['ax'].detach()
                    # prm.data.copy_(optimizer.state[prm]['ax'])

            val_loss2 = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
            print('-' * 89)

            if val_loss2 < stored_loss:
                # model_save(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                #            vocabulary, val_loss2, math.exp(val_loss2), vars(args), epoch)
                model_state_save(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                           vocabulary, val_loss2, math.exp(val_loss2), vars(args), epoch)
                print('Saving Averaged!')
                stored_loss = val_loss2

            # nparams = 0
            # nparams_in_temp_keys = 0
            for prm in model.parameters():
                # nparams += 1
                if prm in tmp.keys():
                    # nparams_in_temp_keys += 1
                    # prm.data = tmp[prm].clone()
                    prm.data = tmp[prm].detach()
                    prm.requires_grad = True
            # print('params {}, params in tmp keys: {}'.format(nparams, nparams_in_temp_keys))
            del tmp
        else:
            print('{} model params (SGD before eval)'.format(len([prm for prm in model.parameters()])))
            val_loss = evaluate(val_data, eval_batch_size)
            print('{} model params (SGD after eval)'.format(len([prm for prm in model.parameters()])))
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)

            if val_loss < stored_loss:
                # model_save(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                #            vocabulary, val_loss, math.exp(val_loss), vars(args), epoch)
                model_state_save(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                           vocabulary, val_loss, math.exp(val_loss), vars(args), epoch)
                print('Saving model (new best validation)')
                stored_loss = val_loss

            if args.asgd:
                if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                        len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                # if 't0' not in optimizer.param_groups[0]:
                    print('Switching to ASGD')
                    # optimizer = ASGD(trainable_parameters, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
                    optimizer = ASGD(params, lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)

            if epoch in args.when:
                print('Saving model before learning rate decreased')
                # model_save('{}.e{}'.format(os.path.join(CKPT_DIR, args.save), model, criterion, optimizer,
                #            vocabulary, val_loss, math.exp(val_loss), vars(args), epoch))
                model_state_save('{}.e{}'.format(os.path.join(CKPT_DIR, args.save), args.save), model, criterion, optimizer,
                           vocabulary, val_loss, math.exp(val_loss), vars(args), epoch)
                print('Dividing learning rate by 10')
                optimizer.param_groups[0]['lr'] /= 10.

            best_val_loss.append(val_loss)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_load(os.path.join(CKPT_DIR, args.save))

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
