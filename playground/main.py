rubbishStr = 'hahahahahahahahahahaha'
retriever.insertContext(rubbishStr)
answer_result = pipe.run("What is XP-936?",params={
    "Retriever": {
        "top_k": 1
    },
    "Reranker": {
        "top_k": 1
    },
    "generation_kwargs":{
        "do_sample": False,
        "max_new_tokens": 128
    }
})
print(f"Answer: {answer_result['answers'][0].answer}")

for i in strList:
    print(i)

for i in strList:
    retriever.insertContext(i)

answer_result = pipe.run("What is XP-936?",params={
    "Retriever": {
        "top_k": 2
    },
    "Reranker": {
        "top_k": 2
    },
    "generation_kwargs":{
        "do_sample": False,
        "max_new_tokens": 128
    }
})
print(f"Llama: {answer_result['answers'][0].answer}")