import os
import Filesystem as F
import transformers
import Utils
import logging
import StandardLM.TextCorpusReader as TCR
import StandardLM.MiniTransformerXL as TXL

def reading_in_wt2(splitname):
    Utils.init_logging("MyTests-StandardLM-reading_in_wt2.log")
    if splitname == Utils.TRAINING:
        split_fname = F.WT_TRAIN_FILE
    elif splitname == Utils.VALIDATION:
        split_fname = F.WT_VALID_FILE
    elif splitname == Utils.TEST:
        split_fname = F.WT_TEST_FILE

    vocab_filename = "vocabulary_" + F.WT2 + "_" + F.SEMCOR + ".txt"
    vocab_filepath = os.path.join(F.FOLDER_VOCABULARY, vocab_filename)
    tokenizer = transformers.TransfoXLTokenizer(vocab_file=vocab_filepath)

    train_txt_file = open(os.path.join(F.CORPORA_LOCATIONS[F.WT2], split_fname), "r", encoding="utf-8")

    for i in range(10):
        line = train_txt_file.readline()
        if len(line)>0:
            logging.info("Line=" + line)
            ids = tokenizer(line)
            logging.info("ids=" + str(ids))
            tokenized_line = tokenizer.convert_ids_to_tokens(ids["input_ids"])
            logging.info("tokenized_line=" + str(tokenized_line) + "\n***\n")


def loading_encoded_corpus(splitname):
    Utils.init_logging("MyTests-StandardLM-loading_encoded_corpus.log")
    vocab_filename = "vocabulary_" + F.WT2 + "_" + F.SEMCOR + ".txt"
    vocab_filepath = os.path.join(F.FOLDER_VOCABULARY, vocab_filename)
    tokenizer = transformers.TransfoXLTokenizer(vocab_file=vocab_filepath)

    #txt_corpus_fpath, numIDs_fpath = TCR.get_corpus_fpaths(corpus_name=F.WT2,
    #                                                       split=splitname, vocabulary_sources_ls=[F.WT2, F.SEMCOR])

    corpus_chunks_ls = TXL.get_numerical_corpus(corpus_name=F.WT2, split_name=splitname,
                                                vocabulary_sources_ls=[F.WT2, F.SEMCOR], chunk_size=10)
    for i in range(10):
        encoded_fragment = corpus_chunks_ls[i]
        logging.info("encoded_fragment=" + str(encoded_fragment))
        fragment_tokens = tokenizer.convert_ids_to_tokens(encoded_fragment)
        logging.info("fragment_tokens=" + str(fragment_tokens))


