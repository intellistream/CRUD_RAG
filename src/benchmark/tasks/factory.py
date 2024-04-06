from src.benchmark.tasks.continue_writing import ContinueWriting
from src.benchmark.tasks.hallucinated_modified import HalluModified
from src.benchmark.tasks.quest_answer import QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs
from src.benchmark.tasks.summary import Summary


def create_task(args):
    task_mapping = {
        'event_summary': [Summary],
        'continuing_writing': [ContinueWriting],
        'hallu_modified': [HalluModified],
        'quest_answer': [QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs],
        'all': [Summary, ContinueWriting, HalluModified, QuestAnswer1Doc, QuestAnswer2Docs, QuestAnswer3Docs]
    }
    if args.task not in task_mapping:
        raise ValueError(f"Unknown task: {args.task}")
    tasks = [task(use_quest_eval=args.quest_eval, use_bert_score=args.bert_score_eval) for task in
             task_mapping[args.task]]
    return tasks
