from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer


sample = ["En este último caso normalmente es un rey pero también puede ser un príncipe, como el caso de Mónaco, un emperador como en Japón u ostentar otro título nobiliario."]

checkpoint = "/data/enxebre/results/checkpoint-60000"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

sample_tokenized = tokenizer(sample, return_tensors="pt").input_ids

print(sample_tokenized)

outputs = model.generate(sample_tokenized)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
