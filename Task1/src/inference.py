import os
import argparse
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


# Load constants from .env file
load_dotenv()
MODEL_LOCAL_PATH = os.getenv('MODEL_LOCAL_PATH')
LOGGING_LEVEL = os.getenv('LOGGING_LEVEL')
LOGGING_FORMAT = os.getenv('LOGGING_FORMAT')
LOGGING_DATE_FORMAT = os.getenv('LOGGING_DATE_FORMAT')


def print_result(result):
    '''
    Print the result of the NER pipeline with formatting.

    Args:
        result (list): The result of the NER pipeline.
    '''
    for entity in result:
        print(f"Word: {entity['word']}, Start: {entity['start']}, End: {entity['end']}")
        print(f"Entity: {entity['entity']}, Score: {entity['score']}")
        print()
        

def main(example):
    logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT, datefmt=LOGGING_DATE_FORMAT)

    model = AutoModelForTokenClassification.from_pretrained(MODEL_LOCAL_PATH)
    logging.info('Model loaded')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
    logging.info('Tokenizer loaded')

    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)
    logging.info('NER pipeline created')

    result = ner_pipeline(example)
    logging.info(f"Result: {result}")

    print_result(result)


if __name__ == '__main__':
    # Get the example from the command line
    parser = argparse.ArgumentParser(description='Inference script for the NER model.')
    parser.add_argument('example', type=str, help='The example to run the NER model on.')
    args = parser.parse_args()
    example = args.example
    
    main(example)