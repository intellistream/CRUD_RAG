from loguru import logger

from src.benchmark import config_parser
from src.benchmark.datasets import get_task_datasets
from src.benchmark.tasks import tasks_factory
from src.components.pipeline_factory import PipelineFactory
from src.evaluator.evaluator import BaseEvaluator


def main():
    parser = config_parser.get_arg_parser()
    args = parser.parse_args()
    logger.info("Configurations: {}", args)
    pipe = PipelineFactory.create_pipeline(args)
    tasks = tasks_factory.get_task(args)
    datasets = get_task_datasets(args.data_path, args.task)
    logger.info("Evaluation started.")
    evaluator = BaseEvaluator(args, pipe, tasks, datasets)
    evaluator.run()
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
