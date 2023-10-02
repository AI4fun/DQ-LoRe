import evaluate
import logging
import re
logger = logging.getLogger(__name__)

pattern = r"###\s*(-?\d+)"
all_pattern = r'(-?\d+(\.\d+)?)\D*$'
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
            g.append(int(gold['Answer']))
        #golds = [int(re.search(pattern, gold["answer"]).group(1)) for gold in golds]
        print("here=====================")
        print("g:{}".format(g))
        print("preds:{}".format(preds))
        metric = evaluate.load("accuracy")
        p = []
        correct = 0
        sum = 0
        for i in range(len(g)):
            match = re.search(pattern, preds[i].replace(",", ""))
            if match:
                sum += 1
                tmp_num = int(float(match.group(1)))
                p.append(tmp_num)
                if tmp_num == g[i]:
                    correct += 1
            else:
        
                all_m = re.search(all_pattern, preds[i].replace(",", ""))
                if all_m:
                    sum += 1
                    tmp_num = int(float(all_m.group(1)))
                    p.append(tmp_num)
                    if tmp_num == g[i]:
                        correct += 1
                    else:
                        print("i:{} g[i]:{} tmp_num:{} preds[i]:{}".format(i,g[i],tmp_num,preds[i]))
        print("p:{}".format(p))
        print("correct/sum:{}".format(correct/sum))
        #result = metric.compute(references=g, predictions=p)
        result = float(correct) / sum
        print("=========acc result:{}".format(result))
        #assert 1==0
        return result

