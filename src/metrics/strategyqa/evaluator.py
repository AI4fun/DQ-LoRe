import evaluate
import logging
import re
logger = logging.getLogger(__name__)

def check_sentence(sentence):
    # 使用正则表达式查找 "yes" 或 "no"
    matches = re.findall(r'\b(?:yes|no)\b', sentence, flags=re.IGNORECASE)

    if not matches:
        return 0

    # 获取最后一个匹配的单词（'yes' 或 'no'）
    last_match = matches[-1]

    # 判断最后一个单词是 'yes' 还是 'no'
    if last_match.lower() == 'yes':
        return 1
    elif last_match.lower() == 'no':
        return 0
    
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
            g.append(check_sentence(gold["answer"]))
        #golds = [int(re.search(pattern, gold["answer"]).group(1)) for gold in golds]
        print("here=====================")
        print("g:{}".format(g))
        print("preds:{}".format(preds))
        metric = evaluate.load("accuracy")
        p = []
        correct = 0
        for i in range(len(g)):
            p.append(check_sentence(preds[i]))
        for i in range(len(g)):
            if g[i]==p[i]:
                correct += 1
        print("p:{}".format(p))
        print("correct/sum:{}".format(correct/len(g)))
        #result = metric.compute(references=g, predictions=p)
        result = float(correct) / sum
        print("=========acc result:{}".format(result))
        #assert 1==0
        return result

