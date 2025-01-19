import os
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from datasets import load_from_disk
import evaluate
import numpy as np


# Load constants from .env file
load_dotenv()
LOCAL_DATASET_PATH = os.getenv('LOCAL_DATASET_PATH')
MODEL_HUGGINGFACE_NAME = os.getenv('MODEL_HUGGINGFACE_NAME')
MODEL_LOCAL_PATH = os.getenv('MODEL_LOCAL_PATH')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')
EPOCHS = int(os.getenv('EPOCHS'))
LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH')


def tokenize_and_align_labels(examples, tokenizer):
    '''
    Tokenize the examples and align the labels with the tokenized inputs.

    Args:
        examples (dict): The examples to tokenize.
        tokenizer (transformers.Tokenizer): The tokenizer to use.
    '''
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # -100 is the index to ignore.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                temp_label = label[word_idx]
                if temp_label % 2 == 1: # If B-MOUNT is repeated, change it to I-MOUNT.
                    temp_label += 1
                label_ids.append(temp_label)

            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics_extra_args(metric, id2label):
    '''
    Compute the metrics for the model.

    Outer Args:
        metric (evaluate.Metric): The metric to use.
        id2label (dict): The mapping of the label ids to the labels.

    Inner Args:
        eval_preds (tuple): The predictions from the model (logits, labels).

    Returns:
        dict: The metrics for the model (precision, recall, f1, accuracy).
    ''' 
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index and convert ids to labels
        true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": all_metrics["overall_precision"],
            "recall": all_metrics["overall_recall"],
            "f1": all_metrics["overall_f1"],
            "accuracy": all_metrics["overall_accuracy"],
        }
    return compute_metrics


def main():   
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    dataset = load_from_disk(LOCAL_DATASET_PATH)
    logging.info('Dataset loaded')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_HUGGINGFACE_NAME)
    logging.info('Tokenizer loaded')

    tokenized_ds = dataset.map(tokenize_and_align_labels, 
                      batched=True,
                      remove_columns=dataset['train'].column_names,
                      fn_kwargs={'tokenizer': tokenizer})
    logging.info('Dataset tokenized')

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    logging.info('Data collator created')

    # Creating maps for the labels and ids
    label_names = ['O', 'B-MOUNT', 'I-MOUNT']
    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {label: i for i, label in label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(MODEL_HUGGINGFACE_NAME, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    logging.info('Model loaded')

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_PATH,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_strategy='no',
    )

    metric = evaluate.load("seqeval")

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics_extra_args(metric, id2label)
    )
    logging.info('Trainer created')

    trainer.train()
    logging.info('Model trained')

    res = trainer.evaluate(tokenized_ds["test"])
    logging.info('Model evaluated')
    logging.info(res)

    trainer.save_model(MODEL_LOCAL_PATH)
    logging.info('Model saved')

    
if __name__ == '__main__':
    main()
