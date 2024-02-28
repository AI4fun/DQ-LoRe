import evaluate
import logging
import re
logger = logging.getLogger(__name__)

def check_sentence(sentence):
    matches = re.findall(r'\b(?:yes|no)\b', sentence, flags=re.IGNORECASE)
    if not matches:
        return 0
    last_match = matches[-1]
    if last_match.lower() == 'yes':
        return 1
    elif last_match.lower() == 'no':
        return 0
    
class EvaluateTool(object):
    def __init__(self):
        pass

    def evaluate(self, preds, golds):
        
        print("preds:{}".format(preds))
        g = []
        for gold in golds:
            g.append(check_sentence(gold["answer"]))
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
        result = float(correct) / sum
        print("=========acc result:{}".format(result))
        #assert 1==0
        return result

