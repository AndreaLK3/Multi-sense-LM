import transformers
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
import os
import logging
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

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
    def __init__(self, tokenizer, file_path='train', block_size=512):
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


def mask_last_token(inputs, tokenizer):
    """ Mask the last token preparing inputs/ and abels for standard language modeling. """
    labels = inputs.clone()
    rows = labels.shape[0]
    columns = labels.shape[1]
    mask_formasking = torch.tensor([[False if col != columns - 1 else True for col in range(columns)] for row in range(rows)])
    labels[~mask_formasking] = -1  # We only compute loss on masked tokens
    print(labels)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    inputs[mask_formasking] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


class ReducingDistilBERT:

    def __init__(self, ):
        # Constructing the architecture: DistilBERT + last linear layer L_768to300
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.core_model = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        # The model is set in evaluation mode by default. Set it back in training mode with model.train()
        self.core_model.train()

        self.model = nn.Sequential(self.core_model,
                                   nn.Linear(BERT_EMBEDDINGS_DIMENSIONS,TARGET_DIMENSIONS),
                                   nn.Softmax())


    def train(self):
        def train(args, train_dataset, model, tokenizer):
            """ Train the model """
            tb_writer = SummaryWriter()

            args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
            train_sampler = RandomSampler(train_dataset) # cutting off the distributed backend of args.local_rank == 0
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

            # cutting off the choice of max_steps >0
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

            # multi-gpu training (should be after apex fp16 initialization)
            if args.n_gpu > 1:
                model = torch.nn.DataParallel(model)

            # Train!
            logging.info("***** Running training *****")
            logging.info("  Num examples = %d", len(train_dataset))
            logging.info("  Num Epochs = %d", args.num_train_epochs)
            logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
            logging.info("  Total train batch size = %d",
                        args.train_batch_size * args.gradient_accumulation_steps)
            logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
            logging.info("  Total optimization steps = %d", t_total)

            global_step = 0
            tr_loss, logging_loss = 0.0, 0.0
            model.zero_grad()
            train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True) # Must see doc: trange

            for _ in train_iterator:
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
                for step, batch in enumerate(epoch_iterator):
                    inputs, labels = mask_last_token(batch, tokenizer)
                    inputs = inputs.to(args.device)
                    labels = labels.to(args.device)
                    model.train()
                    outputs = model(inputs, masked_lm_labels=labels) #if args.mlm else model(inputs, labels=labels)
                    loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    tr_loss += loss.item()
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        model.zero_grad()
                        global_step += 1

                        if args.local_rank in [-1,
                                               0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                            # Log metrics
                            if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                                results = evaluate(args, model, tokenizer)
                                for key, value in results.items():
                                    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                            tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                            logging_loss = tr_loss

                        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                            checkpoint_prefix = 'checkpoint'
                            # Save model checkpoint
                            output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model,
                                                                    'module') else model  # Take care of distributed/parallel training
                            model_to_save.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                            logging.info("Saving model checkpoint to %s", output_dir)

                            _rotate_checkpoints(args, checkpoint_prefix)

                    if args.max_steps > 0 and global_step > args.max_steps:
                        epoch_iterator.close()
                        break
                if args.max_steps > 0 and global_step > args.max_steps:
                    train_iterator.close()
                    break

            if args.local_rank in [-1, 0]:
                tb_writer.close()

            return global_step, tr_loss / global_step

    def forward(self, x):
        return self.model(x)

