import evaluate
metric = evaluate.load("accuracy")
g = [1,2,3,4,5]
p = [1,2,2,2,2]
result = metric.compute(references=g, predictions=p)
print(result)