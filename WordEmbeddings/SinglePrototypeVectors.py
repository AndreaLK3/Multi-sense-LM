import WordEmbeddings.EmbedWithDBERT as EDB
import WordEmbeddings.EmbedWithFastText as EFT
import pandas as pd
import gensim.models.fasttext as FastText
import os
import Filesystem as F

def compute_single_prototype_embeddings(vocabulary_h5_filepath, sp_out_archive):
    vocab_df = pd.read_hdf(vocabulary_h5_filepath, mode='r')

    fasttext_fpath = os.path.join(F.FOLDER_WORD_EMBEDDINGS, F.FASTTEXT_EMBEDDINGS_FILE)
    fasttext_vectors = FastText.load_facebook_vectors(fasttext_fpath)

    for idx_word_freq_tpl in vocab_df.itertuples():
        word_vector = fasttext_vectors[idx_word_freq_tpl[1]]


