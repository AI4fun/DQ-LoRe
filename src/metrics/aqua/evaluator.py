import evaluate
import logging
import re
logger = logging.getLogger(__name__)

pattern = r'[A-Z](?=\))'
class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        #match = re.search(pattern, golds)
        #for gold in golds:
        #    print(gold["answer"])
        #    print("############:{}".format(int(re.search(pattern, gold["answer"]).group(1))))
        
        print("preds:{}".format(preds))
        g = []
        for gold in golds:
            g.append(gold['correct'])
        #golds = [int(re.search(pattern, gold["answer"]).group(1)) for gold in golds]
        print("here=====================")
        print("g:{}".format(g))
        print("preds:{}".format(preds))
        metric = evaluate.load("accuracy")
        p = []
        correct = 0
        sum = 0
        for i in range(len(preds)):
            matches = re.findall(pattern, preds[i])
            if matches:
                tmp = matches[-1]
        
            else:
                #print(f"{i} not find")
                A_matches = re.findall(r'[A-Z]', preds[i])
                tmp = A_matches[-1]
        p.append(tmp)
        print("p:{}".format(p))
        #print("correct/sum:{}".format(correct/sum))
        #result = metric.compute(references=g, predictions=p)
        result = float(correct) / len(g)
        print("=========acc result:{}".format(result))
        #assert 1==0
        return result

