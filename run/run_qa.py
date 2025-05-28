import argparse
from datetime import datetime
import functools
import json
import os
import time

from dataset import *
from inference import *
from utils.eval import evaluate_predictions
from utils.logger import *
from utils.utils import *


def parse_key_value(arg):
    """Parses key=value string into a (key, value) pair, converting value to int/float if needed."""
    if '=' not in arg:
        raise argparse.ArgumentTypeError(
            "Arguments must be in key=value format")
    key, value = arg.split('=', 1)
    try:
        # Try to cast to int or float
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  # Keep as string if it can't be converted
    return key, value


async def generate_prediction(id: str = "",
                              query: str = "",
                              query_time: datetime = None,
                              ans: str = "",
                              logger: BaseProgressLogger = DefaultProgressLogger(),
                              **kwargs):
    start_time = time.perf_counter()

    prediction = await participant_model.generate_answer(query=query, query_time=query_time, **kwargs)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    logger.add_stat({
        "id": id,
        "query": query,
        "query_time": query_time,
        "ans": ans,
        "prediction": prediction,
        "processing_time": round(elapsed_time, 2)
    })
    print(len(logger.processed_questions))
    logger.update_progress({"last_question_total": round(elapsed_time, 2)})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        required=True, help="Evaluation dataset")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_MAP.keys(), help="Model to run inference with")
    parser.add_argument("--num-workers", type=int, default=128,
                        help="Number of workers generating the answers")
    parser.add_argument("--queue-size", type=int, default=128,
                        help="Queue size of data loading")
    parser.add_argument("--split", type=int, default=0,
                        help="Dataset split index")
    parser.add_argument("--prefix", type=str,
                        help="Prefix added to the result file name")
    parser.add_argument("--postfix", type=str,
                        help="Postfix added to the result file name")
    parser.add_argument("--keep", action='store_true',
                        help="Keep the progress file")
    parser.add_argument('--config', nargs='*', type=parse_key_value,
                        help="Override model config as key=value")
    args = parser.parse_args()

    config = {
        "num_workers": args.num_workers,
        "queue_size": args.queue_size,
        "split": args.split,
    }

    progress_path = f"results/{args.model}{f"_{args.prefix}" if args.prefix else ""}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
    result_path = f"results/{args.model}{f"_{args.prefix}" if args.prefix else ""}_{args.dataset}_results{f"_{args.postfix}" if args.postfix else ""}.json"
    logger = QAProgressLogger(progress_path=progress_path)
    print(logger.processed_questions)

    if args.dataset.lower() == "movie":
        domain = "movie"
        loader = MovieDatasetLoader(
            os.path.join(DATASET_PATH, "crag_task_1_and_2_dev_v4.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "sports":
        domain = "sports"
        loader = SportsDatasetLoader(
            os.path.join(DATASET_PATH, "crag_task_1_and_2_dev_v4.jsonl.bz2"),
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "multitq":
        domain = "open"
        loader = MultiTQDatasetLoader(
            "dataset/MultiTQ",
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    elif args.dataset.lower() == "timequestions":
        domain = "yearly question"
        loader = TimeQuestionsDatasetLoader(
            "dataset/TimeQuestions",
            config, "qa", logger,
            processor=functools.partial(generate_prediction, logger=logger)
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")

    participant_model = MODEL_MAP[args.model](
        domain=domain,
        config=dict(args.config) if args.config else None,
        logger=logger
    )

    loop = always_get_an_event_loop()
    loop.run_until_complete(
        loader.run()
    )

    inf_token_usage = token_counter.get_token_usage()
    token_counter.reset_token_usage()
    results = [
        {"id": int(stat["id"]), "query": stat["query"], "query_time": stat["query_time"],
         "ans": stat["ans"], "prediction": stat["prediction"], "processing_time": stat["processing_time"]}
        for stat in logger.progress_data["stats"]
    ]
    results = sorted(results, key=lambda x: x["id"])
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    queries = [item["query"] for item in results]
    ground_truths_list = [[str(item["ans"])] for item in results]
    predictions = [str(item["prediction"]) for item in results]

    stats, history = evaluate_predictions(
        queries, ground_truths_list, predictions, 'llama', batch_size=64
    )
    eval_token_usage = token_counter.get_token_usage()
    stats.update({
        "inf_prompt_tokens": inf_token_usage.get("prompt_tokens"),
        "inf_completion_tokens": inf_token_usage.get("completion_tokens"),
        "inf_total_tokens": inf_token_usage.get("total_tokens"),
        "eval_prompt_tokens": eval_token_usage.get("prompt_tokens"),
        "eval_completion_tokens": eval_token_usage.get("completion_tokens"),
        "eval_total_tokens": eval_token_usage.get("total_tokens")
    })
    for idx in range(len(results)):
        id = results[idx]['id']
        results[idx]['score'] = history[idx]['score']
        results[idx]['explanation'] = history[idx]['explanation']
    results.insert(0, stats)
    # Save to a JSON file
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    if not args.keep:
        os.remove(progress_path)

    logger.info(
        f"Done inference in {args.dataset} dataset on {args.model}_model ✅")
    logger.info(
        f"Inference token usage: {inf_token_usage}; Eval token usage: {eval_token_usage}")
