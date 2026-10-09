"""Optional paid model evaluations; never represented as browser benchmark scores."""
import argparse
import json
import os
from pathlib import Path
from app import pipeline

parser=argparse.ArgumentParser()
parser.add_argument('--framework',choices=['ragas','deepeval'],required=True)
args=parser.parse_args()
if not os.getenv('OPENAI_API_KEY'):
    raise SystemExit('Set OPENAI_API_KEY to run model-judged evaluation (API costs apply).')
questions=['Can I get a refund for my annual subscription?','How do I fix SSO_403 sign-in error?','API returns 429. How long should I wait to retry?']
rows=[]
for q in questions:
    r=pipeline.ask(q)
    rows.append({'user_input':q,'response':r['answer'],'retrieved_contexts':[s['text'] for s in r['sources']]})
if args.framework=='ragas':
    from ragas import EvaluationDataset,evaluate
    from ragas.metrics import Faithfulness
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI
    result=evaluate(EvaluationDataset.from_list(rows),metrics=[Faithfulness()],llm=LangchainLLMWrapper(ChatOpenAI(model=os.getenv('JUDGE_MODEL','gpt-4.1-mini'))))
    result.to_pandas().to_json('ragas-results.json',orient='records',indent=2)
    print(result)
else:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    scores=[]
    for r in rows:
        metric=FaithfulnessMetric(threshold=.8,model=os.getenv('JUDGE_MODEL','gpt-4.1-mini'),include_reason=True)
        metric.measure(LLMTestCase(input=r['user_input'],actual_output=r['response'],retrieval_context=r['retrieved_contexts']))
        scores.append({'question':r['user_input'],'score':metric.score,'reason':metric.reason})
    Path('deepeval-results.json').write_text(json.dumps(scores,indent=2))
    print(json.dumps(scores,indent=2))
