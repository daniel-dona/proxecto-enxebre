from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


metric = load_metric("sacrebleu")

dataset = load_dataset("daniel-dona/paracrawl-9-gl-es")

dataset = dataset["train"].train_test_split(test_size=0.001)

tokenizer = AutoTokenizer.from_pretrained("t5-small")


def preprocess_function(examples):
	
	prefix = "translate Spanish to Galician: "
		
	inputs = [prefix + example for example in examples["es"]]
	targets = [example for example in examples["gl"]]

	model_inputs = tokenizer(inputs)

	with tokenizer.as_target_tokenizer():
		labels = tokenizer(targets)

	model_inputs["labels"] = labels["input_ids"]
	
	return model_inputs

def postprocess_text(preds, labels):
	preds = [pred.strip() for pred in preds]
	labels = [[label.strip()] for label in labels]

	return preds, labels

def compute_metrics(eval_preds):
	preds, labels = eval_preds
	
	'''if isinstance(preds, tuple):
		preds = preds[0]
		
	if isinstance(labels, tuple):
		labels = labels[0]'''
		
	#print(eval_preds)
		
	decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

	# Replace -100 in the labels as we can't decode them.
	labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
	
	decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

	# Some simple post-processing
	decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

	result = metric.compute(predictions=decoded_preds, references=decoded_labels)
	result = {"bleu": result["score"]}

	prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
	
	result["gen_len"] = np.mean(prediction_lens)
	
	result = {k: round(v, 4) for k, v in result.items()}
	
	return result

tokenized_dataset = dataset.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
	output_dir="./results",
	learning_rate=1e-5,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=1,
	weight_decay=0.01,
	save_total_limit=5,
	num_train_epochs=10,
	report_to="tensorboard",
	optim="adamw_torch",
#	do_predict=True,
	save_steps=25000,
#	evaluation_strategy="steps",
#	eval_steps=1000,
	fp16=True,
	fp16_full_eval=True,
	eval_accumulation_steps=1,
)

trainer = Seq2SeqTrainer(
	model=model,
	args=training_args,
	train_dataset=tokenized_dataset["train"],
	eval_dataset=tokenized_dataset["test"],
	tokenizer=tokenizer,
	data_collator=data_collator,
#	compute_metrics=compute_metrics
)

trainer.train()

