import json
import os
from abc import ABC
from argparse import ArgumentParser
from threading import Lock

from haystack import Pipeline, Document
from loguru import logger
from tqdm import tqdm

from src.benchmark.tasks.base import BaseTask

def filter_data(data):
    """skip data of type(Document)"""
    if isinstance(data, dict):
        return {k: filter_data(v) for k, v in data.items() if not isinstance(v, Document)}
    elif isinstance(data, list):
        return [filter_data(item) for item in data if not isinstance(item, Document)]
    else:
        return None if isinstance(data, Document) else data


class BaseEvaluator(ABC):
    def __init__(self, args: ArgumentParser, pipe: Pipeline, task: BaseTask, dataset: list[dict],
                 output_dir: str = 'output'):
        """
        Initializes the evaluator with the provided Haystack pipeline.

        Args:
            args: Configuration arguments for the evaluator.
            pipe (Pipeline): The Haystack pipeline for document retrieval and processing.
            task (BaseTask): The evaluation task.
            dataset (list[dict]): The dataset for evaluation.
            output_dir (str): The directory for result output and caching.
        """
        self.pipe = pipe
        self.args = args
        self.dataset = dataset
        self.task = task
        self.lock = Lock()
        self.model_name = args.model_name_or_path
        collection_name = args.collection_name
        output_dir = os.path.join(output_dir, f'{collection_name}_{args.model_name_or_path}')

        self.output_path = os.path.join(
            output_dir, f'{self.task.__class__.__name__}_{args.model_name_or_path}.json'
        )
        dir_path = os.path.dirname(self.output_path)
        if not (os.path.exists(dir_path) and os.path.isdir(dir_path)):
            os.makedirs(dir_path)

    def task_generation(self, data_point):
        """
        Processes a single data point through the Haystack pipeline.

        Args:
            data_point (dict): A single data point from the dataset.

        Returns:
            The result of processing the data point through the Haystack pipeline.
        """
        try:
            # Ensure thread safety if using multithreading
            with self.lock:
                # Running the Haystack pipeline with the provided data point
                result = self.pipe.run(query=data_point['event'])
                return result

        except Exception as e:
            logger.warning(f"Processing failed for data_point with error: {repr(e)}")
            return None

    def batch_scoring(self, datasets: list[dict], sort=True, show_progress_bar=False, contain_original_data=False):
        """Perform batch scoring on the given dataset.

        Args:
            dataset (list[dict]): The dataset for evaluation.
            sort (bool): Whether to sort the results by id.
            show_progress_bar (bool): Whether to display a progress bar.

        Returns:
            list[dict]: List of results.
        """

        if os.path.exists(self.output_path):  # Resume evaluation
            results = self.read_output().get('results', [])
            results = self.remove_invalid(results)
            saved_ids = [result['id'] for result in results]
        else:
            results = []
            saved_ids = []

        for dataset in datasets:
            for task in self.task:
                for data_point in (tqdm(dataset, desc=self.model_name) if show_progress_bar else dataset):
                    if data_point['ID'] in saved_ids:
                        continue  # Skip results that have already been evaluated and are valid
                    try:
                        generated_text = self.task_generation(data_point)
                        data_point["generated_text"] = generated_text
                        result = {'id': data_point['ID'], **task.scoring(data_point)}
                        if contain_original_data:
                            result['original_data'] = data_point
                        results.append(result)
                    except Exception as e:
                        logger.warning(repr(e))
                        raise
        return sorted(results, key=lambda x: x['id']) if sort else results

    def save_output(self, output: dict) -> None:
        """Save evaluation results."""
        output = filter_data(output)
        # for i in range(len(output['results'])):
        #     del output['results'][i]['original_data']['generated_text']['documents']

        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)

    def read_output(self) -> dict:
        with open(self.output_path) as f:
            return json.load(f)

    def run(self, sort=True, show_progress_bar=True, contain_original_data=True) -> dict:
        info = {
            'task': self.task.__class__.__name__,
        }
        results = self.batch_scoring(self.dataset, sort, show_progress_bar, contain_original_data)
        valid_results = self.remove_invalid(results)
        overall = []
        try:
            for task in self.task:
                overall.append(task.compute_overall(valid_results) if len(valid_results) > 0 else {}) \
                    # 保存用于评估的RAGQuestEval QA问答对
                if task.use_quest_eval:
                    self.lock.acquire()
                    task.quest_eval.save_quest_gt(self.task.__class__.__name__)
                    self.lock.release()

        except Exception as e:
            logger.warning(repr(e))
            overall.append(dict())

        self.save_output(output := {'info': info, 'overall': overall, 'results': results})
        print(f'Output saved at {self.output_path}!')
        return output

    @staticmethod
    def remove_invalid(results: list[dict]) -> list[dict]:
        """Remove invalid results from the list and return the cleaned results."""
        return [result for result in results if result['valid']]
