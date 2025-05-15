import trafilatura
from typing import AsyncGenerator, Any, Dict, List

from utils.data import *
from utils.logger import *
from utils.utils import parse_timestamp

class MultiTQDatasetLoader(BaseDatasetLoader):
    
    def __init__(self,
                 data_path: str, 
                 config: Dict[str, Any], 
                 mode: str = "doc",
                 logger: BaseProgressLogger = DefaultProgressLogger(),
                 **kwargs):
        super().__init__(config, mode, **kwargs)

        self.data_path = data_path
        self.logger = logger
        self.split = config.get("split", 0)
        self.question_path = os.path.join(data_path, f"questions/test_1000_split{self.split}.json")
        with open(self.question_path, 'r', encoding='utf-8') as file:
                self.data = json.load(file)

        self.data_generator = iter(self.data)
    
    async def load_doc(self) -> AsyncGenerator[Dict[str, Any], None]:
        raise NotImplementedError("MultiTQ dataset is only used for inference experiment.")
    
    async def load_query(self) -> AsyncGenerator[Dict[str, Any], None]:
        while True:
            try:
                item = next(self.data_generator)
            except StopIteration:
                break

            query_id = item.get("quid", None)
            if not query_id:
                continue

            if query_id in self.logger.processed_questions:
                continue

            query = item.get("question", "")
            ans = item.get("answers", [])
            query_time = item.get("query_time", None)

            if query_time:
                query_time = parse_timestamp(query_time)
            else:
                query_time = None  # Or simulate with a default timestamp

            yield {
                "id": query_id,
                "query": query,
                "query_time": query_time,
                "ans": ans
            }