#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
#import openai
import csv
import json
import random
import argparse
import string
def extract_subqs(subqs):
    final = []
    res = []
    final1 = []
    res1 = []
    for line in subqs.strip().split('\n'):
        if line.strip() == '':
            continue

        if line.strip().lower().startswith('decomposed subquestion'):
            final.append(line.strip())
        else:
            res.append(line.strip())
            if '?' in line:
                final1.append(line.strip())
            else:
                res1.append(line.strip())

    if final == []:
        #print(subqs)
        if final1 == []:
            print('------------------')
            exit(0)
        else:
            return '\n'.join(final1), '\n'.join(res1)
    return '\n'.join(final), '\n'.join(res)

# for only sft qa based on qd and prediction via gpt, option b
def zeroshot_qdqagpt_add(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    #subqs, res = extract_subqs(row["subqs"])

    if '?' in row["subqs"]:
        data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "")})
    else:
        data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["pred"]})
    content.append(data)

# for option b1
def b1(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    if '?' in row["subqs"]:
        data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. The answer to the original question is supposed to be attached in the end in the format of \'The answer is: \'. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "")})
    else:
        data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["pred"]})
    content.append(data)

# for only sft qa based on qd, option b2
def b2(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    if '?' in row["subqs"]:
        data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. The answer to the original question is supposed to be attached in the end in the format of \'The answer is: \'. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "")})
        pred = row["pred"].strip()
        if 'The answer is: ' in pred:
            final_pred = "The answer is: ".join(pred.split("The answer is: ")[:-1]) + "The answer is: " + row["Golden"]
        else:
            final_pred = "\n".join(pred.split("\n")[:-1]) + "\nThe answer is: " + row["Golden"]
        data["conversations"].append({"from": "gpt", "value": final_pred})
    else:
        data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
        data["conversations"].append({"from": "gpt", "value": row["Golden"]})
    content.append(data)
    #data = {"id": "identify_" + str(count), "conversations": []}
    ##subqs, res = extract_subqs(row["subqs"])

    #if '?' in row["subqs"]:
    #    data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "")})
    #else:
    #    data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    #data["conversations"].append({"from": "gpt", "value": row["Golden"]})
    #content.append(data)
# for only sft qa based on qd, option g and h
def zeroshot_qdqa_add(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    #subqs, res = extract_subqs(row["subqs"])

    if '?' in row["subqs"]:
        data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "")})
    else:
        data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["Golden"]})
    content.append(data)

# for only sft qa based on qd, option g1 and h1
def g1(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    #subqs, res = extract_subqs(row["subqs"])

    if '?' in row["subqs"]:
        data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["context"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "") + "\n" + row["q_only"]})
    else:
        data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["Golden"]})
    content.append(data)

# for only sft qa based on qd, option g2 and h2
def g2(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    #subqs, res = extract_subqs(row["subqs"])

    if '?' in row["subqs"]:
        data["conversations"].append({"from": "human", "value": """Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"] + '\nSubquestions: ' + row["subqs"].replace("Subquestions: ", "") + "\n" + row["q_only"]})
    else:
        data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["Golden"]})
    content.append(data)

# for fully sft qa, option f
def zeroshot_qa_add(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    #subqs, res = extract_subqs(row["subqs"])
    data["conversations"].append({"from": "human", "value": """Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["Golden"]})
    content.append(data)

def zeroshot_qsubq_add(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    #subqs, res = extract_subqs(row["subqs"])
    data["conversations"].append({"from": "human", "value": """Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than three subquestions are formulated for each question. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. \nContext: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": row["subqs"]})
    content.append(data)

qsubq_demo = """Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
Decomposed subquestion 1: How old was Mohamed four years ago?
Decomposed subquestion 2: How old is Kody?"""

def qsubq_add(content, count, row, demo):
    data = {"id": "identify_" + str(count), "conversations": []}
    subqs, res = extract_subqs(row["subqs"])
    data["conversations"].append({"from": "human", "value": """I want to decompose a complex question into several related subquestions that would help me to answer it.
I will show you an example in how to do it first and then give you the real question.

Example question:""" + demo + """

Now comes with our real question: """ + row["Question"]})
    data["conversations"].append({"from": "gpt", "value": subqs})
    content.append(data)

def cotft_0_add(content, count, row):
    data = {"id": "identify_" + str(count), "conversations": []}
    subqs, res = extract_subqs(row["subqs"])
    data["conversations"].append({"from": "human", "value": """I want to solve a complex question by answering several related subquestions that would help me to answer it first. Now comes with our real question and its subquestions, I want you to answer the subquestions one by one and finally solve the original question: """ + row["Question"] + "\n" + subqs.replace("\n", " ")})
    #print(data["conversations"][-1]["value"])
    #exit(0)
    data["conversations"].append({"from": "gpt", "value": res})
    content.append(data)

cotft_demo = """Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
Decomposed subquestion 1: How old was Mohamed four years ago?
Decomposed subquestion 2: How old is Kody?
Answer 1: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. The answer is 56.
Answer 2: Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32.
So the answer for original complex question is: 32"""

def cotft_1_add(content, count, row, demo):
    data = {"id": "identify_" + str(count), "conversations": []}
    subqs, res = extract_subqs(row["subqs"])
    data["conversations"].append({"from": "human", "value": """I want to solve a complex question by answering several related subquestions that would help me to answer it first. I will show you an example in how to do it first and then give you the real question and the decomposed subquestions.

Example question: """ + demo + """

Now comes with our real question: """ + row["Question"] + "\n" + subqs})

    data["conversations"].append({"from": "gpt", "value": res})
    content.append(data)

def demo_add(content, count, row, demo):
    data = {"id": "identify_" + str(count), "conversations": []}
    data["conversations"].append({"from": "human", "value": """I want to solve a complex question \'""" + row["Question"] + """\' by answering several related subquestions that would help me to answer it first. Can you give me an example?"""})
    data["conversations"].append({"from": "gpt", "value": demo})

    #print(data["conversations"][-1]["value"])
    #exit(0)
    content.append(data)

def remove_space(q):
    #print(q)
    final = []
    qs = [i.strip() for i in q.split('\n')]
    for q in qs:
        if q != '':
            final.append(q)
    #print('\n'.join(final))
    #print('0' * 20)
    return '\n'.join(final)

def answer_json_to_strings(answer: dict[str, any]) -> tuple[tuple[str, ...], str]:
    """
    Takes an answer JSON blob from the DROP data release and converts it into strings used for
    evaluation.
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        if len(answer["spans"]) == 1:
            return tuple(answer["spans"]), "span"
        else:
            return tuple(answer["spans"]), "spans"
#            return tuple([', '.join(answer["spans"])]), "spans"
    elif "date" in answer:
        return (
            tuple(
                [
                    "{0} {1} {2}".format(
                        answer["date"]["day"], answer["date"]["month"], answer["date"]["year"]
                    )
                ]
            ),
            "date",
        )
    else:
        raise ValueError(
            f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}"
        )

def clean(ans):
    if "-" in ans:
        ans = ans.replace("-", " ")
    return ans

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qdpath", type=str, default="golden-decomposing-by-gpt-3.5-turbo/3full_train-cot_decompose_drop_0-1_11068.csv", help="using gpt-3.5-turbo generated data")
    parser.add_argument("--qapath", type=str, default="gpt-3.5-turbo-results/3gpt-3.5-turbo-cot_infer_0_cot_decompose_drop_11068.csv", help="using gpt-3.5-turbo generated data")
    parser.add_argument("--full", action="store_true", help="set if wanna extracting all decomposition no matter true or not")
    parser.add_argument("--ft", action="store_true", help="set if generating data for fine-tuning vicuna as answer generator")
    parser.add_argument("--cotft_0", action="store_true", help="add cot fine-tuning")
    parser.add_argument("--cotft_1", action="store_true", help="add cot fine-tuning")
    parser.add_argument("--demo", action="store_true", help="add demo generation")
    parser.add_argument("--ftall", action="store_true", help="option f, python3 newselect_correct_chain_drop.py --ftall --full")
    parser.add_argument("--qdqagpt", action="store_true", help="option b, python3 newselect_correct_chain_drop.py --qdqagpt --full")
    parser.add_argument("--b1", action="store_true", help="option b1, python3 newselect_correct_chain_drop.py --b1 --full --qapath gpt-3.5-turbo-results/full_train-gpt-3.5-turbo-cot_infer_0_cot_decompose_drop_11068.csv")
    parser.add_argument("--d2", action="store_true", help="option d2, python3 newselect_correct_chain_drop.py --zero_shot --d2")
    parser.add_argument("--qdqa", action="store_true", help="option g, python3 newselect_correct_chain_drop.py --qdqa --full")
    parser.add_argument("--g1", action="store_true", help="option g1 or h1, python3 newselect_correct_chain_drop.py --g1 --full")
    parser.add_argument("--g2", action="store_true", help="option g2 or h2, python3 newselect_correct_chain_drop.py --g2 --full")
    parser.add_argument("--b2", action="store_true", help="option b2, python3 newselect_correct_chain_drop.py --b2 --full --qapath gpt-3.5-turbo-results/full_train-gpt-3.5-turbo-cot_infer_0_cot_decompose_drop_11068.csv")
    # python3 newselect_correct_chain_drop.py --b2 --qapath gpt-3.5-turbo-results/full_train-gpt-3.5-turbo-cot_infer_0_cot_decompose_drop_11068.csv
    # option h, python3 newselect_correct_chain_drop.py --qdqa
    parser.add_argument("--zero_shot", action="store_true", help="set if generating data for zero-shot fine-tuning vicuna as question decomposor")
    parser.add_argument("--random", action="store_true", help="if set, then random sample the demo examples.")
    args = parser.parse_args()

    inf = json.load(open('../drop_dataset_with_test_questions/drop_dataset_train.json'))
    total = 0
    golden = {}
    ori_golden = {}
    question = {}
    q_only = {}
    qd = {}
    context = {}
    for context_id, annotation in inf.items():
        passage = annotation["passage"]
        if passage[-1] not in string.punctuation:
            passage += '.'

        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            
            gold_answer, gold_type = answer_json_to_strings(qa_pair["answer"])
            golden[query_id] = clean(gold_answer[0].lower().strip()) 
            ori_golden[query_id] = clean(gold_answer[0])
            question[query_id] = passage.strip() + "\nQuestion: " + qa_pair["question"].strip()
            context[query_id] = passage.strip()
            q_only[query_id] = "\nQuestion: " + qa_pair["question"].strip()
            #print(gold_answer, gold_type)
            #print('-----' * 10)
            #total += 1
            #if total == 10:
            #    exit(0)

    def vicuna_extract_subqs(subqs):
        final = []
        for line in subqs.strip().split('\n'):
            if '?' in line:
                final.append(line.strip())
            if line == '' and final != []:
                break
        if final == []:
            return 'No decomposition.'
        else:
            ans = '\n'.join(final) + '\n'
            if 'ubquestion' not in ans:
                ans = 'Subquestions: ' + ans
        return ans.replace('\n', ' ')

    print('reading qd')
    with open(args.qdpath) as csvfile:
        cr = csv.DictReader(csvfile)
        for row in cr:
            query_id = row["query_id"]
            print(row["subqs"])
            qd[query_id] = vicuna_extract_subqs(row["subqs"])
            print('==='*10)
            print(qd[query_id])
            print('-----' * 20)
            

    from metrics import get_metrics
    from evaluation import extract_answer
    count, total = 0, 0

    used = {}
    if args.d2:
        with open('gpt-3.5-turbo-results/full_train-gpt-3.5-turbo-cot_infer_0_cot_decompose_drop_11068.csv') as csvfile:
            cr = csv.DictReader(csvfile)
            for row in cr:
                query_id = row["query_id"]
                #pred = extract_answer(row['predict'].lower().strip()) 
                pred = extract_answer(row['predict'].strip()) 
                if args.b1 or args.b2:
                    pred = row['predict']
                used[query_id] = pred

    content = []
    qsubqs = []
    demos = [] 
    rows = []
    print('reading qa')
    with open(args.qapath) as csvfile:
        cr = csv.DictReader(csvfile)
        for row in cr:
            query_id = row["query_id"]
            #pred = extract_answer(row['predict'].lower().strip()) 
            pred = extract_answer(row['predict'].strip()) 
            em_score, f1_score, predicted_bags, gold_bags = get_metrics(pred, row['Golden'].lower().strip())
            if golden[query_id] in pred.lower() or args.full:
                #real_subqs, res = extract_subqs(row["subqs"])

                #judge = False
                #for i in range(len(real_subqs.split('\n'))):
                #    if real_subqs.split('\n')[i].startswith('Decomposed subquestion '+str(i+1)):
                #        continue
                #    judge = True
                #    print(real_subqs)
                #    print('=====' * 20)
                #    break
                #if judge:
                #    continue

                q = remove_space(question[query_id])
                subqs = remove_space(qd[query_id])
                qsubqs.append(q + '\n' + remove_space(qd[query_id]))
                #demos.append(q + '\n' + real_subqs + '\n' + res)
                if args.b1 or args.b2:
                    pred = row['predict']
                rows.append({'Golden': ori_golden[query_id], "subqs": subqs, "Question": q, "pred": pred, "q_only": remove_space(q_only[query_id]), "context": remove_space(context[query_id])})
            elif args.d2 and golden[query_id] in used[query_id].lower():
                q = remove_space(question[query_id])
                subqs = remove_space(qd[query_id])
                qsubqs.append(q + '\n' + remove_space(qd[query_id]))
                #demos.append(q + '\n' + real_subqs + '\n' + res)

                rows.append({'Golden': ori_golden[query_id], "subqs": subqs, "Question": q, "pred": used[query_id], "q_only": remove_space(q_only[query_id]), "context": remove_space(context[query_id])})

            if args.ft:
                data = {"id": "identify_" + str(count), "conversations": []}
                data["conversations"].append({"from": "human", "value": row["Question"]})
                data["conversations"].append({"from": "gpt", "value": row["Golden"]})
                content.append(data)
                count += 1

        print(len(rows))
    print('finish qa reading')

    for row in rows:
        if args.random:
            qsubq_add(content, count, row, qsubqs[random.randint(0, len(qsubqs) - 1)])
        else:
            if args.b1:
                b1(content, count, row)
            elif args.b2:
                b2(content, count, row)
            elif args.ftall:
                zeroshot_qa_add(content, count, row)
            elif args.qdqagpt:
                zeroshot_qdqagpt_add(content, count, row)
            elif args.qdqa:
                zeroshot_qdqa_add(content, count, row)
            elif args.g1:
                g1(content, count, row)
            elif args.g2:
                g2(content, count, row)
            elif args.zero_shot:
                zeroshot_qsubq_add(content, count, row)
            else:
                qsubq_add(content, count, row, qsubq_demo)
        count += 1

        if args.cotft_0:
            cotft_0_add(content, count, row)
            count += 1

        if args.cotft_1:
            if args.random:
                cotft_1_add(content, count, row, demos[random.randint(0, len(demos) - 1)])
            else:
                cotft_1_add(content, count, row, cotft_demo)
            count += 1
    
        if args.demo:
            if args.random:
                demo_add(content, count, row, demos[random.randint(0, len(demos) - 1)])
            else:
                demo_add(content, count, row, cotft_demo)
            count += 1
    prefix = ''
    if args.full:
        prefix += 'full_'
    if args.random:
        prefix += 'random-demo_'
    if args.ft:
        prefix += 'ft_'
    if args.cotft_0:
        prefix += 'cotft_0_'
    if args.cotft_1:
        prefix += 'cotft_1_'
    if args.demo:
        prefix += 'demo_'
    if args.ftall:
        prefix += 'f0shot_2_'
    elif args.qdqagpt:
        prefix += 'b0shot_2_'
    elif args.b1:
        prefix += 'b1_0shot_'
    elif args.d2:
        prefix += 'd2_0shot_'
        # 48853 -> 56450
    elif args.b2:
        prefix += 'b2_0shot_'
    elif args.qdqa:
        if args.full:
            prefix += 'g0shot_2_'
        else:
            prefix += 'h0shot_2_'
    elif args.g1:
        if args.full:
            prefix += 'g1_0shot_2_'
        else:
            prefix += 'h1_0shot_'
    elif args.g2:
        if args.full:
            prefix += 'g2_0shot_2_'
        else:
            prefix += 'h2_0shot_'
    elif args.zero_shot:
        prefix += 'new0shot_'
    else:
        prefix += 'new1shot_'
    print(prefix+"droptrain.json")
    #`json.dump(content, open("./playground/data/" + prefix + "droptrain.json", "w"), indent=2)


