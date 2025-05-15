import argparse
import asyncio

from kg.kg_preprocessor import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Evaluation dataset")
    args = parser.parse_args()

    preprocessors = []
    if args.dataset.lower() == "movie":
        preprocessors.append(MovieKG_Preprocessor())
    elif args.dataset.lower() == "sports":
        preprocessors.append(SoccerKG_Preprocessor())
        preprocessors.append(NBAKG_Preprocessor())
    elif args.dataset.lower() == "multitq":
        preprocessors.append(MultiTQKG_Preprocessor())
    elif args.dataset.lower() == "timequestions":
        preprocessors.append(TimeQuestionsKG_Preprocessor())
    else:
        raise NotImplementedError(f"Dataset {args.dataset} is not supported.")

    # Async Route
    async def main():
        for preprocessor in preprocessors:
            await preprocessor.preprocess()
            await preprocessor.close()

    loop = asyncio.new_event_loop()  # Create a new event loop
    asyncio.set_event_loop(loop)  # Set it as the current loop
    loop.run_until_complete(main())
    
    logger.info("Data imported to Neo4j ✅")
