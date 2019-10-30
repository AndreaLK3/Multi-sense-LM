import transformers
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import logging
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import Filesystem as F
import Utils
import math


BERT_EMBEDDINGS_DIMENSIONS = 768
TARGET_DIMENSIONS = 300

# Objective:    Create a model that will obtain sentence embeddings for Definitions and Examples,
#               where the embeddings will have dimension d=300,
#               thus executing a dimensionality reduction from the d=768 used in the BERT architectures.

# Method:	- add a last linear layer that goes from 768 to 300 on the DistilBERT architecture,
#           - fine-tune the whole instrument on a Language Model task on our chosen training set.
#           - save the model. We will load it and use it as an already-trained instrument in EmbedWithBERT.py

# Part of the code has been modified from: HuggingFace's transformers/examples/run_lm_finetuning.py

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path='train', block_size=510):
        # block_size is 510 not 512, because this is *before* adding [CLS] and [SEP], we would end up out of bounds at 514
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, 'cached_lm_' + str(block_size) + '_' + filename)

        if os.path.exists(cached_features_file):
            logging.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logging.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text)-block_size+1, block_size): # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)

            logging.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, 'wb') as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def mask_last_token(inputs, tokenizer):
    """ Mask the last token preparing inputs and labels for standard language modeling. """
    labels = inputs.clone()
    rows = labels.shape[0]
    columns = labels.shape[1]
    mask_formasking = torch.tensor([[False if col != columns - 1 else True for col in range(columns)] for _row in range(rows)])
    labels[~mask_formasking] = -1  # We only compute loss on masked tokens
    print(labels)

    # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    inputs[mask_formasking] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels



class ModifiedDistilBERT:

    def __init__(self, output_dir=os.path.join(F.FOLDER_WORD_EMBEDDINGS, F.FOLDER_DISTILBERT_MODEL)):
        # Constructing the architecture: DistilBERT + last linear layer L_768to300
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.core_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

        #self.model = nn.Sequential(self.core_model,
        #                           nn.Linear(BERT_EMBEDDINGS_DIMENSIONS,TARGET_DIMENSIONS),
        #                           nn.Softmax())
        self.model = self.core_model

        # Other parameters
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_gpu = torch.cuda.device_count()
        self.block_size = self.tokenizer.max_len_single_sentence

        Utils.init_logging(os.path.join("PrepareGraphInput", "ReduceBERT.log"))


    def train(self, train_txt_path=os.path.join(F.FOLDER_WT2, F.WT_TRAIN_FILE),
              validation_txt_path=os.path.join(F.FOLDER_WT2, F.WT_VALID_FILE), num_train_epochs=10,
              learning_rate = 5e-5, adam_epsilon =1e-8, batch_size=4, logging_steps=50, max_grad_norm=1.0):

        tb_writer = SummaryWriter()

        # """ Train the model """
        train_dataset = TextDataset(self.tokenizer, file_path=train_txt_path)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

        t_total = len(train_dataloader) // 1 * num_train_epochs #default number of gradient accumulation steps = 1

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}, # default weight decay = 0.0
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total) # default warmup_steps = 0

        # if self.fp16: ignoring fp16 optimization in the current version

        # multi-gpu training (should be after apex fp16 initialization)
        if self.n_gpu > 1:
            model = torch.nn.DataParallel(self.model)
        else:
            model = self.model
        model.to(self.device)

        # Train!
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(train_dataset))
        logging.info("  Num Epochs = %d", num_train_epochs)
        logging.info("  Total train batch size = %d", batch_size)
        logging.info("  Gradient Accumulation steps = %d", 1)
        logging.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        old_valid_perplexity = math.inf
        model.zero_grad()
        train_iterator = trange(int(num_train_epochs), desc="Epoch") # , disable=args.local_rank not in [-1, 0])
        # set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration") # , disable=args.local_rank not in [-1, 0])
            for step, batch in enumerate(epoch_iterator):
                inputs, labels = mask_last_token(batch, self.tokenizer) # if args.mlm else (batch, batch)
                # we are finetuning a (Distil)BERT model on Language Modeling --> we use mlm
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                logging.debug("inputs.shape=" + str(inputs.shape))

                model.train() # setting training mode
                outputs = model(inputs, labels) # masked_lm_labels if args.mlm else model(inputs, labels=labels)
                loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                # if self.n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
                # # if args.gradient_accumulation_steps > 1:
                # #    loss = loss / args.gradient_accumulation_steps
                # logging.info("loss.shape=" + str(loss.shape))
                # RuntimeError: grad can be implicitly created only for scalar outputs
                loss = loss.mean()
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % 1 == 0: #args.gradient_accumulation_steps
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

                    if global_step % logging_steps == 0:
                        ##### Log metrics
                        # n.: Only evaluating when not using distributed training, otherwise metrics may not average well
                        results = self.evaluate(model, self.tokenizer, validation_txt_path, self.block_size)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar('loss', (tr_loss - logging_loss) / logging_steps, global_step)
                        logging_loss = tr_loss

                        checkpoint_prefix = 'checkpoint'
                        ##### Save model checkpoint
                        output_dir = os.path.join(self.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                        logging.info("Saving checkpoint to output location: " + str(output_dir))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        # torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logging.info("Saving model checkpoint to %s", output_dir)
                        # _rotate_checkpoints(args, checkpoint_prefix)
                        #### Added: Early stopping
                        if results["perplexity"] > old_valid_perplexity:
                            logging.info("The latest validation perplexity " + str(results["perplexity"] +
                                         " is higher than the previous one " + str(old_valid_perplexity) +
                                         " . Stopping training now."))
                            break
                        else:
                            old_valid_perplexity = results["perplexity"] # and we go on

                #if global_step > args.max_steps: close...
            tb_writer.close()
        return global_step, tr_loss / global_step


    def evaluate(self, model, tokenizer, text_fpath, block_size=510, batch_size=4, prefix=""):
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        eval_output_dir = self.output_dir

        eval_dataset = TextDataset(tokenizer, file_path=text_fpath, block_size=block_size)

        if not os.path.exists(eval_output_dir):
            os.makedirs(eval_output_dir)

        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size)

        # Eval!
        logging.info("***** Running evaluation {} *****".format(prefix))
        logging.info("  Num examples = %d", len(eval_dataset))
        logging.info("  Batch size (spread over all GPUs) = %d", batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            inputs, labels = mask_last_token(batch, tokenizer)  # if args.mlm else (batch, batch)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = model(inputs, labels)  # masked_lm_labels= if args.mlm else model(inputs, labels=labels)
                lm_loss = outputs[0]
                eval_loss += lm_loss.mean().item()
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))

        result = {
            "perplexity": perplexity
        }


        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logging.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        return result
