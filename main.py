import datetime
import json
import os
import time
from tqdm.auto import tqdm
from loguru import logger
from inference import *
from dataset.movie_dataset import load_data_in_batches
from utils.eval import evaluate_predictions
from utils.logger import *
from utils.utils import *

def generate_predictions(dataset_path, participant_model):
    """
    Processes batches of data from a dataset to generate predictions using a model.
    
    Args:
    dataset_path (str): Path to the dataset.
    participant_model (object): UserModel that provides `get_batch_size()` and `batch_generate_answer()` interfaces.
    
    Returns:
    tuple: A tuple containing lists of queries, ground truths, and predictions.
    """
    queries, ground_truths, predictions = [], [], []
    batch_size = participant_model.get_batch_size()

    elapsed_time = 0
    for batch in tqdm(load_data_in_batches(dataset_path, batch_size, domain='movie', start_idx=None), desc="Generating predictions"):
        batch_ground_truths = batch["answer"]  # Remove answers from batch and store them

        start_time = time.perf_counter()
        batch_predictions = participant_model.batch_generate_answer(batch)
        end_time = time.perf_counter()
        elapsed_time += end_time - start_time
        
        queries.extend(batch["query"])
        ground_truths.extend(batch_ground_truths)
        predictions.extend(batch_predictions)
    
    print(f"Elapsed time: {elapsed_time} seconds")
    return queries, ground_truths, predictions

if __name__ == "__main__":
    from inference.user_config import UserModel

    DATA_PATH = os.path.join(DATASET_PATH, "crag_task_1_and_2_dev_v4.jsonl.bz2")
    EVALUATION_MODEL_NAME = os.getenv("EVALUATION_MODEL_NAME", "gpt-4-0125-preview")

    # Generate predictions
    logger = QAProgressLogger(progress_path="results/answer_progress.json")
    participant_model = UserModel(logger=logger)
    queries, ground_truths, predictions = generate_predictions(
        DATA_PATH, participant_model
    )

    # Save to a JSON file
    results = [
        {"query": q, "ground_truth": gt, "prediction": p} #"prediction": p[0]} , "logs": p[1]}
        for q, gt, p in zip(queries, ground_truths, predictions)
    ]
    with open(f"results/{participant_model.name}_movie_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Evaluate Predictions
    with open(f"results/{participant_model.name}_movie_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)  # Deserialize JSON into Python dictionary/list

    queries = [item["query"] for item in results]
    ground_truths_list = [[str(item["ground_truth"])] for item in results]
    predictions = [item["prediction"] for item in results]

    results, history = evaluate_predictions(
        queries, ground_truths_list, predictions, 'llama', batch_size=64
    )

    history.insert(0, results)
    # Save to a JSON file
    with open(f"results/{participant_model.name}_movie_eval.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    logger.info(f"Done inference on {participant_model.name}_model ✅")