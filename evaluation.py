import tensorflow as tf
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForQuestionAnswering
from datasets import load_dataset, load_metric
import time

model_h5_path = "/home/Team99/distilbert-base-uncased-distilled-squad/tf_model.h5"
config_json_path = "/home/Team99/distilbert-base-uncased-distilled-squad/config.json"

model = TFDistilBertForQuestionAnswering.from_pretrained(model_h5_path, config=config_json_path)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
test_datasets = load_dataset("squad")
metric = load_metric("squad")
validation_data = test_datasets['validation']

contexts = [item['context'] for item in validation_data]
questions = [item['question'] for item in validation_data]
answers = [item['answers']['text'][0] for item in validation_data]

def calculation(tokenizer, model, contexts, questions, answers):
    f1_scores = []
    em_scores = []
    total_time = 0.0
    for context, question, answer in zip(contexts, questions, answers):
        inputs = tokenizer(question, context, return_tensors="tf", padding=True, truncation=False)
        start_time = time.time()
        outputs = model(inputs)
        end_time = time.time()
        start_logits, end_logits = outputs.start_logits, outputs.end_logits
        
        start_index = tf.argmax(start_logits, axis=1).numpy()[0]
        end_index = tf.argmax(end_logits, axis=1).numpy()[0]

        predicted_answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index+1])

        # Calculate F1 Score
        f1_score = calculate_f1_score(predicted_answer, answer)
        f1_scores.append(f1_score)
        
        # Calculate EM Score
        em_score = 1 if predicted_answer.strip().lower() == answer.strip().lower() else 0
        em_scores.append(em_score)
        
        # Calculate Inf. Time
        total_time += (end_time - start_time)
    
    return np.mean(f1_scores) * 100, em_scores, total_time

def calculate_f1_score(prediction, true_answer):
    # Split text into word units
    prediction_tokens = prediction.lower().split()
    true_tokens = true_answer.lower().split()

    # Find common_tokens
    common_tokens = set(prediction_tokens) & set(true_tokens)
    
    # Calculate precision & recall
    precision = len(common_tokens) / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
    recall = len(common_tokens) / len(true_tokens) if len(true_tokens) > 0 else 0
    
    # Calculate F1 Score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score

f1_score, em_score, total_time = calculation(tokenizer, model, contexts, questions, answers)
exact_match = sum(em_score) / len(em_score) * 100

print("f1_score :", f1_score)
print("exact_match :", exact_match)
print("time :", total_time)