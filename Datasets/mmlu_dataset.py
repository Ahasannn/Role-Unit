import glob
import os
from pathlib import Path
from typing import Union, List, Literal, Any, Dict, Optional

import numpy as np
import pandas as pd
from abc import ABC

class MMLUDataset(ABC):
    def __init__(self,
        split: Union[Literal['dev'], Literal['val'], Literal['test']],
        data_root: Optional[str] = None,
        stratified_limit: int = 0,
        ) -> None:

        self._split = split

        data_path = self._resolve_data_path(self._split, data_root)
        self._total_df: pd.DataFrame = self._load_data(data_path, stratified_limit)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _resolve_data_path(split: str, data_root: Optional[str]) -> str:
        if data_root:
            root = Path(data_root)
            if root.is_dir() and root.name != split:
                root = root / split
            return f"{root}{os.sep}"

        candidates = [
            Path("Datasets") / "MMLU" / "data" / split,
            Path("datasets") / "MMLU" / "data" / split,
        ]
        for path in candidates:
            if path.is_dir():
                return f"{path}{os.sep}"
        return f"{candidates[0]}{os.sep}"

    @staticmethod
    def _load_data(
        data_path: str,
        stratified_limit: int = 0,
        ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        if stratified_limit > 0:
            # Stratified sampling: take proportional samples from each topic
            print(f"Using stratified sampling to select {stratified_limit} items from all topics...")

            # First pass: load and count items per topic
            topic_data = {}
            total_count = 0
            for path in csv_paths:
                topic_name = os.path.basename(path).replace('.csv', '')
                topic_df = pd.read_csv(path, header=None, names=names, encoding='utf-8')
                topic_data[topic_name] = topic_df
                total_count += len(topic_df)
                print(f"  {topic_name}: {len(topic_df)} items")

            # Calculate proportional samples per topic
            sampled_dfs = []
            remaining = stratified_limit
            topics_sorted = sorted(topic_data.keys())  # Deterministic order

            for i, topic_name in enumerate(topics_sorted):
                topic_df = topic_data[topic_name]
                if i == len(topics_sorted) - 1:
                    # Last topic: take all remaining to reach exactly stratified_limit
                    n_samples = remaining
                else:
                    # Proportional sampling
                    n_samples = int(len(topic_df) / total_count * stratified_limit)

                # Take first n_samples items (deterministic)
                n_samples = min(n_samples, len(topic_df))
                sampled_dfs.append(topic_df.iloc[:n_samples])
                remaining -= n_samples
                print(f"  Sampled {n_samples} from {topic_name}")

            total_df = pd.concat(sampled_dfs, ignore_index=True)
            print(f"Total sampled: {len(total_df)} items")

        else:
            # Original behavior: load all and shuffle
            total_df = pd.DataFrame(columns=names)
            for path in csv_paths:
                single_df = pd.read_csv(path, header=None,
                                names=names,encoding='utf-8')
                total_df = pd.concat([total_df, single_df])

            total_df = total_df.reset_index(drop=True)

            # Pseudorandom shuffle
            total_df = total_df.reindex(rng.permutation(total_df.index))

            print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> Union[pd.DataFrame, pd.Series]:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: Union[pd.DataFrame, pd.Series]) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos+len("answer is"):].strip(":").strip().strip("Option").strip()
            answer = answer[0] # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def record_to_target_answer(record: Union[pd.DataFrame, pd.Series]) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
