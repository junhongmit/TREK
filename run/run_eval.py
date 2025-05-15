import argparse
import json

from dataset.movie_dataset import *
from inference import *
from utils.eval import evaluate_predictions
from utils.logger import *
from utils.utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="movie", help="Evaluation dataset")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reeval", type=str,
                       help="Specify the .json file you want to re-evaluate")
    # choices=MODEL_MAP.keys(),
    group.add_argument("--model", type=str, help="Model to run inference with")
    parser.add_argument("--prefix", type=str,
                        help="Prefix added to the result file name")
    parser.add_argument("--postfix", type=str,
                        help="Postfix added to the result file name")
    args = parser.parse_args()

    other_stat = {}
    if not args.reeval:
        progress_path = f"results/{args.model}{f"_{args.prefix}" if args.prefix else ""}_{args.dataset}_progress{f"_{args.postfix}" if args.postfix else ""}.json"
        result_path = f"results/{args.model}{f"_{args.prefix}" if args.prefix else ""}_{args.dataset}_results{f"_{args.postfix}" if args.postfix else ""}.json"

        logger = QAProgressLogger(progress_path=progress_path)
        if len(logger.progress_data["stats"]) == 0:
            logger.error(f"No progress found for {args.model}_model ❌")
            exit()

        results = [
            {"id": int(stat["id"]), "query": stat["query"], "query_time": stat["query_time"],
             "ans": stat["ans"], "prediction": stat["prediction"], "processing_time": stat["processing_time"]}
            for stat in logger.progress_data["stats"]
        ]
    else:  # Re-evaluate a previous stored results
        with open(args.reeval, "r", encoding="utf-8") as f:
            temp_results = json.load(f)

        results = []
        for result in temp_results:
            if "id" in result:
                results.append(result)
            else:
                other_stat = result
                del other_stat["eval_llm"]

        result_path = args.reeval

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
    stats.update(other_stat)
    stats.update({
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

    logger.info(f"Done inference on {args.model}_model ✅")
    logger.info(f"Token usage: {token_counter.get_token_usage()}")
