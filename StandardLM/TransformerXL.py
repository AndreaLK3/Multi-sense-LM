from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")

model = AutoModel.from_pretrained("transfo-xl-wt103")

ids = tokenizer.encode("The Fulton Country Grand Jury said on Friday that the investigation into Atlanta's recent")
tokenizer.convert_ids_to_tokens(ids)
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)

# Aims: - create a Transformer-XL model, smaller than the version applied on WT-103
#       - train the TXL model on the WikiText-2 dataset. Using: training split. Early stopping on validation, etc.
def txl_on_wt2():
    # get the configuration of the model pre-trained on WT-103
    model_wt103 = AutoModel.from_pretrained("transfo-xl-wt103")
    config = model_wt103.config.copy()

    # apply the necessary modifications to the configuration
    config.n_head = 8       # from: 16
    config.n_layer = 12     # from: 18
    config.mem_len = 800    # from: 1600
    config.d_embed = 512    # from: 1024
    config.d_head = 64      # unchanged
    config.d_inner = 2048   # from: 4096
    config.d_model = 512    # from: 1024

