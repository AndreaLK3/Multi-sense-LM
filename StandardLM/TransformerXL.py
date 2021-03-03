from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("transfo-xl-wt103")

model = AutoModel.from_pretrained("transfo-xl-wt103")

ids = tokenizer.encode("The Fulton Country Grand Jury said on Friday that the investigation into Atlanta's recent")
tokenizer.convert_ids_to_tokens(ids)
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)

#
# 
def txl_on_wt2():