#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import json
import os
import openai
import pandas as pd
import sys
import csv
import argparse

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)


# import from self-ask
import string
import re
import urllib.request, json
#from serpapi import GoogleSearch

prompt_self_ask_4_shot = ['''Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball

Question: Are both the directors of Jaws and Casino Royale from the same country?
Are follow up questions needed here: Yes.
Follow up: Who is the director of Jaws?
Intermediate Answer: The director of Jaws is Steven Spielberg.
Follow up: Where is Steven Spielberg from?
Intermediate Answer: The United States.
Follow up: Who is the director of Casino Royale?
Intermediate Answer: The director of Casino Royale is Martin Campbell.
Follow up: Where is Martin Campbell from?
Intermediate Answer: New Zealand.
So the final answer is: No

Question: ''',
'''
Are follow up questions needed here:''', ]
#print([el.strip() for el in open("../openai_key.txt", "r")][0])
openai.api_key = [el.strip() for el in open("../openai_key.txt", "r")][0]
serpapi_key = ""

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(30))
def completion_with_backoff(clean_train_sen, backbone):
    #return ChatGPT_universal_paraphrase(clean_train_sen)
    #return vinilla_ask(text)
    return call_gpt(clean_train_sen, backbone)


def ChatGPT_universal_paraphrase(text):

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are a linguistic expert on text rewriting."},
            {"role": "user", "content": f"Rewrite the paragraph:'{text}' without changing its original sentiment meaning. The new paragraph should have a similar length but significantly different expression. "},
    ]
    )
    return response['choices'][0]['message']['content']

def get_last_line(generated):
    if '\n' not in generated:
        last_line =  generated
    else:
        last_line = generated.split('\n')[-1]

    return last_line

def extract_question(generated):
    if '\n' not in generated:
        last_line = generated
    else: 
        last_line = generated.split('\n')[-1]

    if 'Follow up:' not in last_line:
        print('we probably should never get here...' + generated)

    if ':' not in last_line:
        after_colon = last_line
    else:
        after_colon = generated.split(':')[-1]
    
    if ' ' == after_colon[0]:
        after_colon = after_colon[1:]
    if '?' != after_colon[-1]:
        print('we probably should never get here...' + generated)

    return after_colon

def promptf(question, prompt, backbone='text-davinci-002', intermediate = "\nIntermediate answer:", followup = "Follow up:", finalans= '\nSo the final answer is:', decompose=False):
    query_len, query_times = 0, 0
    #question = 'What rocket was the first spacecraft that ever approached Uranus launched on?'
    cur_prompt = prompt[0] +  question + prompt[1]

    #print(cur_prompt, end ='')
     
    query_len += len(cur_prompt.split())
    query_times += 1
    ret_text = call_gpt(cur_prompt, backbone, intermediate)
    #print(ret_text)
    #print('-' * 100)

    while followup in get_last_line(ret_text):
        cur_prompt += ret_text
        question = extract_question(ret_text)

        #print(question)
        query_len += len(question.split())
        query_times += 1
        external_answer = call_gpt(question, backbone)
        #print(external_answer)
        #print('-' * 100)
       
        if external_answer is not None:
            cur_prompt += intermediate + ' ' + external_answer + '.'
            #print(cur_prompt)
            query_len += len(cur_prompt.split())
            query_times += 1
            ret_text = call_gpt(cur_prompt, backbone, intermediate)
            #print(ret_text)
            #print('-' * 100)
        else:
            #We only get here in the very rare case that Google returns no answer.
            cur_prompt += intermediate
            query_len += len(cur_prompt.split())
            query_times += 1
            gpt_answer = call_gpt(cur_prompt, backbone, ['\n'+followup, finalans])
            cur_prompt += gpt_answer

    if finalans not in ret_text:
        if backbone == 'gpt-3.5-turbo' or backbone == 'gpt-4':
            cur_prompt = prompt[0] +  question
        else:
            cur_prompt += ' ' + ret_text + finalans

        if len(cur_prompt) > 2048:
            cur_prompt = cur_prompt[:2048]

        query_len += len(cur_prompt.split())
        query_times += 1
        ret_text = call_gpt(cur_prompt, backbone, '\n')

    #print('...')
    #print(cur_prompt + ret_text)
    #print('...')
    #exit(0)
    if decompose:
        return ret_text

    return cur_prompt + ret_text, query_len, query_times

def call_gpt(cur_prompt, backbone='text-davinci-002', stop=None):
    if backbone =='gpt-3.5-turbo':
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "user", "content": cur_prompt},
            ]
        )
        return response['choices'][0]['message']['content']
    elif backbone == 'text-davinci-002':
        #print('-' * 50)
        #print(cur_prompt)
        #print('*' * 20)
        ans = openai.Completion.create(
                model="text-davinci-002",
                max_tokens=2048,
                stop=stop,
                prompt=cur_prompt,
                temperature=0)
        returned = ans['choices'][0]['text']
        #print(returned)
        #print( greenify(returned), end='')
    elif backbone == 'text-davinci-003':
        ans = openai.Completion.create(
                model="text-davinci-003",
                max_tokens=2048,
                stop=stop,
                prompt=cur_prompt,
                temperature=0)
        returned = ans['choices'][0]['text']
    elif backbone == 'gpt-4':
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{cur_prompt}"},
            ]
        )
        return response['choices'][0]['message']['content']

    return returned

def call_vicuna(prompt, backbone):
    model, tokenizer = backbone[0], backbone[1]
    prompt = f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"""
    #print(prompt)
    #inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    input_ids = tokenizer([prompt]).input_ids
    tokens = model.generate(
        torch.as_tensor(input_ids).cuda(),
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

def call_batch_vicuna(prompts, backbone):
    import time
    a = time.time()
    model, tokenizer = backbone[0], backbone[1]
    inputs = [f"""A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {prompt} ASSISTANT:"""for prompt in prompts]
    input_ids = tokenizer(inputs, padding=True, max_length=2048, return_tensors="pt", return_attention_mask=True, truncation=True).input_ids
    gen_output = model.generate(
        torch.as_tensor(input_ids).cuda(),
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        temperature=0.7,
        max_new_tokens=1024,
        top_p=1.0,
    )
    seq_length = len(gen_output["scores"])

    # get only the generated text (excluding prompt)
    gen_tokens = gen_output["sequences"][:, -seq_length:]

    gen_texts = [tokenizer.decode(
        output, skip_special_tokens=True)
        for output in gen_tokens.tolist()]
    #print(gen_texts)
    print(time.time() - a)
    return gen_texts


def vinilla_ask(context, question, backbone):
    if isinstance(backbone, str):
        #prefix_prompt = '''Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
        #cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip()
        cur_prompt = context + ' ' + question
        ans = call_gpt(cur_prompt, backbone)
    else:
        ans = call_vicuna(context + ' ' + question, backbone)

    return ans

def extract_answer(generated):
    if '\n' not in generated:
        last_line = generated
    else:
        last_line = generated.split('\n')[-1]

    if ':' not in last_line:
        after_colon = last_line.strip()
    else:
        after_colon = last_line.split(':')[-1].strip()

    if len(after_colon) == 0:
        #print('----empty case----')
        #print(generated)
        return ''

    if '.' == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon

def extract_question(generated):
    if '\n' not in generated:
        last_line =  generated
    else:
        last_line = generated.split('\n')[-1]

    if 'Follow up:' not in last_line:
      print('we probably should never get here...' + generated)

    if ':' not in last_line:
        after_colon = last_line.strip()
    else:
        after_colon = last_line.split(':')[-1].strip()

    if '?' != after_colon[-1]:
        print('we probably should never get here...' + generated)

    return after_colon

def get_last_line(generated):
    if '\n' not in generated:
        last_line =  generated
    else:
        last_line = generated.split('\n')[-1]

    return last_line

def greenify(input):
  return "\x1b[102m" + input + "\x1b[0m"

def yellowfy(input):
  return "\x1b[106m" + input + "\x1b[0m"

prompt_self_ask_gsm8k_1_shot = [
'''Question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
Are follow up questions needed here: Yes.
Follow up: How old was Mohamed four years ago?
Intermediate Answer: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. The answer is 56.
Follow up: How old is Kody?
Intermediate Answer: Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32.
So the final answer is: 32

Question: ''',
'''\nAre follow up questions needed here:''']

def self_ask(question, backbone):
    print(question)
    ret, query_len, query_times = promptf(question, prompt_self_ask_4_shot, backbone)
    #ret = promptf(question, prompt_self_ask_gsm8k_1_shot, backbone)

    print(ret)
    print('=' * 100)
    clean_ans = extract_answer(ret)
    #print(clean_ans)
    return clean_ans, query_len, query_times 


prompt_least_to_most_1_shot = ['''Q: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together? 
A: Let us break down this problem: 1. How many apples does Anna have? 2. How many apples do Elsa and Anna have together?
1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.
2. Elsa and Anna have 5 + 7 = 12 apples together.''',
'''A: Let us break down this problem:''', ]

def least_to_most_1_shot(question, backbone):
    #print(question)
    cur_prompt = prompt_least_to_most_1_shot[0] + '\n\n' + 'Q: ' + question.strip() + '\n' + prompt_least_to_most_1_shot[1]
    #print(cur_prompt)
    ret_text = call_gpt(cur_prompt, backbone)
    #print(ret_text)
    cur_prompt += ret_text + '\n\nThe answer is: '
    #print('*' * 50)
    #print(cur_prompt)
    ret_text = call_gpt(cur_prompt, backbone)
    print('---------')
    print(cur_prompt + ret_text)

    return ret_text.strip()

prompt_least_to_most_11_shot = ['''Q: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
A: To answer the question “How old is Kody?”, we need to know: “How old is Mohamed?”, “How old was Mohamed four years ago?”, “How old was Kody four years ago?”.

Q: If Pam is currently twice as young as Rena is, and in 10 years Rena will be 5 years older than her, how old is Pam now?
A: To answer the question “How old is Pam now?”, we need to know: “How much older is Rena than Pam currently?”.

Q: As a freelancer, Baylor is paid for every finished work of a client he does on a freelance marketplace. Currently, he has $4000 on his dashboard from previous work done. He is currently working for three clients, with the first client paying him half the amount of money he currently has on his dashboard once the job is done. The second client will pay him 2/5 times more money than the first client once Baylor finishes his work. The third client will pay him twice the amount of money the first and second clients pay him together once he finishes the job. How much money will Baylor have in his dashboard after all the clients pay him for his work?
A: To answer the question “How much money will Baylor have in his dashboard after all the clients pay him for his work?”, we need to know: “How much will Baylor’s first client pay him for his work?”, “How much more will Baylor’s second client pay him for his work compared to the first client?”, “How much will Baylor’s second client pay him for his work?”, “How much will the first and second clients pay him together once he finishes the job?”, “How much will Baylor’s third client pay him for his work?”, “How much money will all the clients pay Baylor for his work?”.

Q: Cappuccinos cost $2, iced teas cost $3, cafe lattes cost $1.5 and espressos cost $1 each. Sandy orders some drinks for herself and some friends. She orders three cappuccinos, two iced teas, two cafe lattes, and two espressos. How much change does she receive back for a twenty-dollar bill?
A: To answer the question “How much change does she receive back for a twenty-dollar bill?”, we need to know: “How much did the cappuccinos cost in total?”, “How much did the iced teas cost in total?”, “How much did the cafe lattes cost in total?”, “How much did the espressos cost in total?”, “How much did the drinks cost in total?”.

Q: Betty & Paige are raising money for their kids’ little league team by hosting a bake sale. Betty has baked 4 dozen chocolate chip cookies, 6 dozen oatmeal raisin cookies and 2 dozen regular brownies. Paige baked 6 dozen sugar cookies, 3 dozen blondies and 5 dozen cream cheese swirled brownies. If they sell the cookies for $1.00 apiece and the blondies/brownies at $2.00 apiece, how much money will they raise?
A: To answer the question “How much money will they raise?”, we need to know: “How many dozen cookies did they bake (not including blondies/brownies)?”, “How many cookies did they bake (not including blondies/brownies)?”, “How many dozen blondies/brownies did they bake (not including cookies)?”, “How many blondies/brownies did they bake (not including cookies)?”, “How much money will they raise from the cookies (not including blondies/brownies)?”, “How much money will they raise from the blondies/brownies (not including cookies)?”.

Q: On a moonless night, three fireflies danced in the evening breeze. They were joined by four less than a dozen more fireflies, before two of the fireflies flew away. How many fireflies remained?
A: To answer the question “How many fireflies remained?”, we need to know: “How many fireflies joined?”.

Q: Sam, Sid, and Steve brought popsicle sticks for their group activity in their Art class. Sam has thrice as many as Sid, and Sid has twice as many as Steve. If Steve has 12 popsicle sticks, how many popsicle sticks can they use for their Art class activity?
A: To answer the question “How many popsicle sticks can they use for their Art class activity?”, we need to know: “How many popsicle sticks does Sid have?”, “How many popsicle sticks does Sam have?”.
''']

prompt_1_shot = ['''Q: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together? 
A: Anna has 2 more apples than Elsa, so Anna has 2 + 5 = 7 apples. Elsa and Anna have 5 + 7 = 12 apples together. The answer is 12.''']

def cot_1_shot(question, backbone):
    cur_prompt = prompt_1_shot[0] + '\n\n' + 'Q: ' + question.strip() + '\n' + 'A: '
    ret_text = call_gpt(cur_prompt, backbone)
    return ret_text

prompt_4_shot = ['''Q: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
A: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32.

Q: Carla bought 2 bags of mini peanut butter cups on clearance. Each bag was $6.00 but was 75% off. How much did she spend on 2 bags of candy?
A: Each bag was $6.00 but was 75% off. So each bag cost $6.00 * (1 - 0.75) = $6.00 * 0.25 = $1.50. Carla bought 2 bags. So she spent $1.50 * 2 = $3.00. The answer is 3.

Q: If Pam is currently twice as young as Rena is, and in 10 years Rena will be 5 years older than her, how old is Pam now?
A: Since Rena will be 5 years older than Pam in 10 years, she must be 5 years older than Pam now as well. If Pam is currently twice as young as Rena, that means that Rena is currently twice as old as Pam is. So if P stands for Pam’s age now and R stands for Rena’s age now, then we know that R = 2 * P And since Rena is 5 years older than Pam now, we know that R = P + 5. By substitution, we have P + 5 = 2 * P, which means that P = 5. The answer is 5.

Q: Cappuccinos cost $2, iced teas cost $3, cafe lattes cost $1.5 and espressos cost $1 each. Sandy orders some drinks for herself and some friends. She orders three cappuccinos, two iced teas, two cafe lattes, and two espressos. How much change does she receive back for a twenty-dollar bill?
A: Sandy ordered three cappuccinos, which cost $2 each, so she spent $2 * 3 = $6 on cappuccinos. She ordered two iced teas, which cost $3 each, so she spent $3 * 2 = $6 dollars on ice teas. She ordered two cafe lattes, which cost $1.5 each, so she spent $1.5 * 2 = $3 on cafe lattes. She ordered two espressos, which cost $1 each, so she spent $1 * 2 = $2 on espressos. So altogether, Sandy spent $6 + $6 + $3 + $2 = $17 on drinks, which means that sandy will get $20 - $17 = $3 as change. The answer is 3.''']

def cot_4_shot(question, backbone):
    cur_prompt = prompt_4_shot[0] + '\n\n' + 'Q: ' + question.strip() + '\n' + 'A: '
    ret_text = call_gpt(cur_prompt, backbone)
    return ret_text

def decompose(question, backbone):
    ret_text = call_gpt(question, backbone)
    return ' '.join(ret_text.strip().split('\n')[-1].split()[1:])

def decompose2(question, backbone):
    ret_text = call_gpt(question, backbone)
    return ret_text.strip()

def decompose3(question, subquestion, backbone):
    if subquestion != '':
        cur_prompt = question.strip() + '''\n\nQ''' + subquestion.strip()
    else:
        cur_prompt = question.strip()

    print(cur_prompt)
    print('---cur_prompt---')
    ret_text = call_gpt(cur_prompt, backbone)
    #print('-' * 30)
    #print(ret_text)
    #print('*' * 50)
    return ' '.join(ret_text.strip().split('\n'))

# for option f
def gsm_direct_qa(context, question, subqs, backbone):
    prefix_prompt = '''Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text. Question: '''
    cur_prompt = prefix_prompt + context + ' ' + question.strip()
    #cur_prompt = context + ' ' + question
    print(cur_prompt)
    if isinstance(backbone, str):
        ret_text = call_gpt(cur_prompt, backbone)
    else:
        ret_text = call_vicuna(cur_prompt.strip(), backbone)
    print(ret_text)
    #print('1' * 100)
    return ret_text

# for option f
def direct_qa(context, question, subqs, backbone):
    prefix_prompt = '''Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
    cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip()
    print(cur_prompt)
    if isinstance(backbone, str):
        ret_text = call_gpt(cur_prompt, backbone)
    else:
        ret_text = call_vicuna(cur_prompt.strip(), backbone)
    print(ret_text)
    #print('1' * 100)
    return ret_text

# for option b, g, h
def sft_qa(context, question, subqs, backbone):
    if subqs != '':
        prefix_prompt = '''Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
        cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip() + '\nSubquestions: ' + ' '.join(subqs.split('\n'))
    else:
        prefix_prompt = '''Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
        cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip()
    #print(cur_prompt)
    if isinstance(backbone, str):
        ret_text = call_gpt(cur_prompt, backbone)
    else:
        ret_text = call_vicuna(cur_prompt.strip(), backbone)
    #print(ret_text)
    #print('1' * 100)
    return ret_text

def gsm_cot_infer_0(question, subqs, backbone):
    prefix_prompt = '''I want to solve a complex question by answering several related subquestions that would help me to answer it first. I want you to answer the subquestions one by one and finally solve the original question. The final answer is supposed to attached in the end in the format of \'The answer is: \'. Now comes with our real question and its subquestions: '''
    #if subqs != '':
    #    cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip() + '\nSubquestions: ' + ' '.join(subqs.split('\n'))
    #else:
    #    prefix_prompt = '''Given the provided question, please provide a concise answer. If the answer contains a number, provide only the numerical value, without any accompanying text.\nQuestion: '''

    if isinstance(backbone, str):
        if isinstance(question, str):
            cur_prompt = prefix_prompt + question.strip() + ' ' + ' '.join(subqs.split('\n'))
        else:
            cur_prompt = [prefix_prompt + question[idx].strip() + ' ' + ' '.join(subqs[idx].split('\n')) for idx in range(len(question))]
        print(cur_prompt)

        #ret_text = call_gpt(cur_prompt, backbone)
        ret_text = ''
        return ret_text, len(cur_prompt.split())
    else:
        if isinstance(question, str):
            cur_prompt = prefix_prompt + question.strip() + ' ' + ' '.join(subqs.split('\n'))
            ret_text = call_vicuna(cur_prompt.strip(), backbone)
        else:
            cur_prompt = [prefix_prompt + question[idx].strip() + ' ' + ' '.join(subqs[idx].split('\n')).strip() for idx in range(len(question))]
            ret_text = call_batch_vicuna(cur_prompt, backbone)

    return ret_text

def cot_infer_0(context, question, subqs, backbone):
    #prefix_prompt = '''I want to solve a complex question by answering several related subquestions that would help me to answer it first. I want you to answer the subquestions one by one and finally solve the original question. The final answer is supposed to be attached in the end in the format of \'The answer is: \'. If the answer is numerical, give me the number instead of textual format. Now comes with our real question and its subquestions: '''
    # for exact match
    # 0
    #prefix_prompt = '''Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to the final answer. Please answer each subquestion in a sequential manner and then provide a comprehensive response to the original question. The final answer is supposed to be attached in the end in the format of \'The answer is: \'. If the answer is numerical, give me the number instead of textual format.\nContext: '''
    # for f1 score
    # 1
    #prefix_prompt = '''Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to the final answer. Please answer each subquestion in a sequential manner and then provide a concise answer to the original question. The final answer is supposed to be less than five words and be attached in the end in the format of \'The answer is: \'. If the answer contains a number, give me the number without any text.\nContext: '''
    # for f1 score
    # 2
    #prefix_prompt = '''Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to the final answer. Please address each subquestion sequentially and then concisely respond to the original question using no more than five words. The final answer is supposed to be attached in the end in the format of \'The answer is: \'. If the answer contains a number, give me the number without any text.\nContext: '''
    # 3
    #prefix_prompt = '''Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to the final answer. Please address each subquestion sequentially and then concisely respond to the original question using no more than five words. Conclude with the final answer, formatted as \'The answer is: \'. If the final answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
    # 4
    prefix_prompt = '''Your task is to tackle a complex question by first addressing a series of related subquestions, which will collectively guide you to solve the original question. Please provide a concise answer to the original question with no more than five words. The answer to the original question is supposed to be attached in the end in the format of \'The answer is: \'. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
    
    if isinstance(backbone, str):
        if isinstance(question, str):
            #cur_prompt = prefix_prompt + question.strip() + ' ' + ' '.join(subqs.split('\n'))
            if subqs != '':
                cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip() + '\nSubquestions: ' + ' '.join(subqs.split('\n'))
            else:
                prefix_prompt = '''Given the provided context and question, please provide a concise answer with no more than five words. If the answer contains a number, provide only the numerical value, without any accompanying text.\nContext: '''
                cur_prompt = prefix_prompt + context + '\nQuestion: ' + question.strip()
                #cur_prompt = context + ' ' + question.strip()
            print(cur_prompt)
        else:
            cur_prompt = [prefix_prompt + question[idx].strip() + ' ' + ' '.join(subqs[idx].split('\n')) for idx in range(len(question))]

        ret_text = call_gpt(cur_prompt, backbone)
    else:
        if isinstance(question, str):
            cur_prompt = prefix_prompt + question.strip() + ' ' + ' '.join(subqs.split('\n'))
            ret_text = call_vicuna(cur_prompt.strip(), backbone)
        else:

            cur_prompt = [prefix_prompt + question[idx].strip() + ' ' + ' '.join(subqs[idx].split('\n')).strip() for idx in range(len(question))]
            ret_text = call_batch_vicuna(cur_prompt, backbone)

    return ret_text

def cot_infer_1(question, subqs, backbone):
    if isinstance(backbone, str):
        prefix_prompt = '''I want to solve a complex question by answering several related subquestions that would help me to answer it first. The final answer is supposed to attached in the end in the format of \'The answer is: \'. 
I will show you an example in how to do it first and then give you the real question and the decomposed subquestions.
Example question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
Decomposed subquestion 1: How old was Mohamed four years ago?
Decomposed subquestion 2: How old is Kody?
Answer 1: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. The answer is 56.
Answer 2: Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32.
The answer is: 32.
Now comes with our real question: '''
        if isinstance(question, str):
            cur_prompt = prefix_prompt + question.strip() + '\n' + subqs + '\n'
        else:
            cur_prompt = [prefix_prompt + question[idx].strip() + ' ' + ' '.join(subqs[idx].split('\n')) for idx in range(len(question))]

        ret_text = call_gpt(cur_prompt, backbone)
    else:
        prefix_prompt = '''I want to solve a complex question by answering several related subquestions that would help me to answer it first. The final answer is supposed to attached in the end in the format of \'The answer is: \'. I will show you an example in how to do it first and then give you the real question and the decomposed subquestions. Example question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody? Decomposed subquestion 1: How old was Mohamed four years ago? Decomposed subquestion 2: How old is Kody? Answer 1: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. The answer is 56. Answer 2: Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32. The answer is: 32. Now comes with our real question: '''
        if isinstance(question, str):
            cur_prompt = prefix_prompt + question.strip() + ' ' + ' '.join(subqs.split('\n'))
            ret_text = call_vicuna(cur_prompt.strip(), backbone)
        else:
            cur_prompt = [prefix_prompt + question[idx].strip() + ' ' + ' '.join(subqs[idx].split('\n')).strip() for idx in range(len(question))]
            ret_text = call_batch_vicuna(cur_prompt, backbone)

    return ret_text

def cot_infer_cot(question, subqs, backbone):
    cur_prompt = '''I want to solve a complex question by answering several related subquestions that would help me to answer it first.
I will show you an example in how to do it first and then give you the real question and the decomposed subquestions.

Example question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
Decomposed subquestion 1: How old was Mohamed four years ago?
Answer 1: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. The answer is 56.
Decomposed subquestion 2: How old is Kody?
Answer 2: Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32.
So the answer for original complex question is: 32

Now comes with our real question and its subquestions, I want you to answer the subquestions one by one and finally solve the original question: ''' + question.strip() + '\n' + subqs + '\n'

    ret_text = call_gpt(cur_prompt, backbone)

    return ret_text

def vinilla_vicuna(question, subqs, backbone):
    cur_prompt = '''I want to solve a complex question by answering several related subquestions that would help me to answer it first.
Now comes with our real question and its subquestions, I want you to answer the subquestions one by one and finally solve the original question: ''' + question.strip() + '\n' + subqs + '\n'

    ret_text = call_vicuna(cur_prompt, backbone)
    return ret_text

def run_gsm8k(model, modelname, datasetname, seed, backbone, subset=-1, recover_num=-1):
    query_len, query_times = 0, 0
    if isinstance(backbone, str):
        backbone_name = backbone
    else:
        backbone_name = backbone[2]

    if subset != -1:
        pathname = backbone_name+'-results/'+modelname+'_'+datasetname+'_'+str(subset)+'_'+seed+'.csv'
    else:
        pathname = backbone_name+'-results/'+modelname+'_'+datasetname+'_'+seed+'.csv'

    fieldnames = ['Question', 'Golden', 'predict']
    if os.path.exists(pathname):
        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
        print(pathname)
    else:
        outfile = open(pathname, 'w')

        print('start from scratch')
        print(pathname)
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

    count = 0
    tp = 0
    with open('./datasets/gsm8k/test.jsonl') as inf:
        for line in inf:
            if count >= recover_num:
                data = json.loads(line)
                q, a = data['question'], data['answer'].split('####')[-1].replace(',', '').strip()
                pred, tmp_query_len, tmp_query_times = model(q, backbone)
                pred = pred.replace(',', '').strip()

                query_len += tmp_query_len 
                query_times += tmp_query_times
                print('golden: ', a)
                print('pred: ', pred)
                print('query_len: ', tmp_query_len)
                print('query_times: ', tmp_query_times)
                print('------------------')

                count += 1
                if a in extract_answer(pred.lower().strip()):
                    tp += 1
                print(f"{tp} / {count} = {tp / count}")

                writer.writerow({'Question': q, 'Golden': a, 'predict': pred if isinstance(backbone, str) else pred.split('Assistant:')[1]})
            print(count)
            if count == subset:
                break
    print(query_len)
    print(query_times)
    outfile.close()

def extract_self_ask_subqs(ret):
    a = ret.split('Follow up: ')
    b = ''
    for i in range(1, len(a)):
        b += f'Decomposed subquestion {i}: ' + a[i].split('\n')[0] + '\n'

    return b

def self_ask_decompose(question, backbone):
    #ret = promptf(question, prompt_self_ask_4_shot, backbone)
    ret = promptf(question, prompt_self_ask_gsm8k_1_shot, backbone, decompose=True)

    return extract_self_ask_subqs(ret)

def gsm_cot_decompose(question, backbone):
    #seed=1
    #cur_prompt = '''Please decompose this complex question into several related subquestions that would help me to answer it: '''
    #seed=2
    #cur_prompt = '''I want to decompose a complex question into related subquestions that would help me to answer it. The context and the question will be given. If you believe the question can be answered without decomposition, return \"No decomposition\"; otherwise, return the decomposed subquestions. Do not return the decomposed subquestions if answering them won\'t help answer the original question.\nContext: '''
    #seed=4
    #cur_prompt = '''I want to decompose a complex question into most related subquestions that would help me to answer it. The context and the question will be given. If you believe the question can be answered without decomposition, return \"No decomposition\"; otherwise, return the decomposed subquestions. Do not return the decomposed subquestions if answering them won\'t help or even harm the original question answering.\nContext: '''
    #seed=5
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. Avoid repeating similar subquestions.\nContext: '''
    #seed=6
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Please ensure that each subquestion directly contributes to a comprehensive understanding and answer of the main question. Avoid repeating similar subquestions.\nContext: '''
    #seed=7
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than four subquestions are formulated for each question. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. \nContext: '''
    #seed=8
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than three subquestions are formulated for each question. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful.\nQuestion: '''
    seed=9
    cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than three subquestions are formulated for each question. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful.\nQuestion: '''
    #if not isinstance(backbone, str) and '0shot' in backbone[2]:
    #    cur_prompt = '''Please decompose this complex question into several related subquestions that would help me to answer it: '''

    if isinstance(question, str):
        cur_prompts = cur_prompt + question.strip()
    else:
        cur_prompts = [cur_prompt + q.strip() for q in question]

    if isinstance(backbone, str):
        query_len = len(cur_prompts.split())
        ret_text = ''
        #ret_text = call_gpt(cur_prompts, backbone)
    else:
        if isinstance(question, str):
            query_len = len(cur_prompts.split())
            ret_text = call_vicuna(cur_prompts, backbone)
        else:
            query_len = sum([len(i.split()) for i in cur_prompts])
            ret_text = call_batch_vicuna(cur_prompts, backbone)
    return ret_text, query_len

def cot_decompose(context, question, backbone):
    no_context = False
    if context == '':
        if '. ' in question:
            context = '. '.join(question.split('. ')[:-1]) + '.'
            question = question.split('. ')[-1]
        else:
            no_context = True
    #    cur_prompt = '''I want to decompose a complex question into several related subquestions that would help me to answer it.
    #I will show you an example in how to do it first and then give you the real question.
    #
    #Example question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
    #Decomposed subquestion 1: How old was Mohamed four years ago?
    #Answer 1: We were told that Mohamed is currently twice 30 years old, so he is currently 30 * 2 = 60 years old. That means that four years ago he must have been 60 - 4 = 56 years old. The answer is 56.
    #Decomposed subquestion 2: How old is Kody?
    #Answer 2: Four years ago, Kody was half as old as Mohamed, so Kody must have been 56 / 2 = 28 years old then. Since Kody was 28 years old four years ago, she must now be 28 + 4 = 32 years old. The answer is 32.
    #So the answer for original complex question is: 32
    #
    #Now comes with our real question: '''
    #if not isinstance(backbone, str) and '0shot' in backbone[2]:
    #    cur_prompt = '''Please decompose this complex question into several related subquestions that would help me to answer it: '''

    #seed=1
    #cur_prompt = '''Please decompose this complex question into several related subquestions that would help me to answer it: '''
    #seed=2
    #cur_prompt = '''I want to decompose a complex question into related subquestions that would help me to answer it. The context and the question will be given. If you believe the question can be answered without decomposition, return \"No decomposition\"; otherwise, return the decomposed subquestions. Do not return the decomposed subquestions if answering them won\'t help answer the original question.\nContext: '''
    #seed=4
    #cur_prompt = '''I want to decompose a complex question into most related subquestions that would help me to answer it. The context and the question will be given. If you believe the question can be answered without decomposition, return \"No decomposition\"; otherwise, return the decomposed subquestions. Do not return the decomposed subquestions if answering them won\'t help or even harm the original question answering.\nContext: '''
    #seed=5
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. Avoid repeating similar subquestions.\nContext: '''
    #seed=6
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Please ensure that each subquestion directly contributes to a comprehensive understanding and answer of the main question. Avoid repeating similar subquestions.\nContext: '''
    #seed=7
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than four subquestions are formulated for each question. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. \nContext: '''
    seed=8
    cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than three subquestions are formulated for each question. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. \nContext: '''
    #seed=9
    #cur_prompt = '''Your task is to break down a given complex question into the most relevant and helpful subquestions, ensuring that no more than two subquestions are formulated for each question. Both the context and the main question will be provided to you. If the question does not need breaking down to be answered, return \"No decomposition\"; otherwise, list the necessary subquestions. Only return subquestions that directly aid in answering the original question, avoiding any that could be harmful or unhelpful. \nContext: '''
    if no_context:
        cur_prompt = cur_prompt.replace('\nContext: ', '')

    if isinstance(question, str):
        cur_prompts = cur_prompt + context.strip() + "\nQuestion: " + question.strip()
    else:
        cur_prompts = [cur_prompt + q.strip() for q in question]

    print(cur_prompts)
    if isinstance(backbone, str):
        ret_text = call_gpt(cur_prompts, backbone)
    else:
        if isinstance(question, str):
            ret_text = call_vicuna(cur_prompts, backbone)
        else:
            ret_text = call_batch_vicuna(cur_prompts, backbone)
    return ret_text

def cot_qd(question, backbone):
    cur_prompt = '''I want to decompose a complex question into several related subquestions that would help me to answer it. Each decomposed subquestion is supposed to start with \"Decomposed subquestion\".
I will show you an example in how to do it first and then give you the real question.

Example question: Four years ago, Kody was only half as old as Mohamed. If Mohamed is currently twice 30 years old, how old is Kody?
Decomposed subquestion 1: How old was Mohamed four years ago?
Decomposed subquestion 2: How old is Kody?

Now comes with our real question: ''' + question.strip() + ''' Please decompose it into several related subquestions with no duplication.'''
    if isinstance(backbone, str):
        ret_text = call_gpt(cur_prompt, backbone)
    else:
        ret_text = call_vicuna(cur_prompt, backbone)
    return ret_text


def direct_decompose(question, backbone):
    cur_prompt = 'Can you help me decompose this complex question into several subquestions: ' + question.strip()
    if isinstance(backbone, str):
        ret_text = call_gpt(cur_prompt, backbone)
    else:
        ret_text = call_vicuna(cur_prompt, backbone)
    return ret_text

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

def decompose_drop(model, modelname, datasetname, seed, backbone, subset=-1, whether_train=False, numerical_only=False, gpu_id=0, total_gpu=8):
    if whether_train:
        inf = json.load(open('../drop_dataset_with_test_questions/drop_dataset_train.json'))
        total = 77409
        if numerical_only:
            total = 46973
    else:
        inf = json.load(open('../drop_dataset_with_test_questions/drop_dataset_dev.json'))
        total = 9536
        #if numerical_only:
        #    total = 19793
    if subset != -1:
        total = subset

    each_gpu_num = total // total_gpu
    st = gpu_id * each_gpu_num
    en = (gpu_id + 1) * each_gpu_num
    print(st, en)
    if gpu_id + 1 == total_gpu:
        en = total

    if isinstance(backbone, str):
        backbone_name = backbone
    else:
        backbone_name = backbone[2]

    if subset == -1:
        ph = 'full_'
    else:
        ph = str(subset) + '_'

    if not os.path.exists('golden-decomposing-by-' + backbone_name+'/'):
        os.mkdir('golden-decomposing-by-' + backbone_name+'/')  

    if whether_train:
        pathname = 'golden-decomposing-by-' + backbone_name+'/'+ph+'train-'+modelname+'_'+datasetname+'_'+str(gpu_id)+'-'+str(total_gpu)+'_'+seed+'.csv'
    else:
        pathname = 'golden-decomposing-by-' + backbone_name+'/'+ph+'dev-'+modelname+'_'+datasetname+'_'+str(gpu_id)+'-'+str(total_gpu)+'_'+seed+'.csv'

    recover_num = 0
    used = set()
    fieldnames = ['query_id', 'subqs']
    if os.path.exists(pathname):
        with open(pathname) as csvfile:
            cr = csv.DictReader(csvfile)

            for row in cr:
                #print(row)
                #if row['query_id'] not in used:
                used.add(row['query_id'])
                recover_num += 1
        print('gpu' + str(gpu_id) + ' recover_num: ' + str(recover_num))
        print('left: ' + str(total - len(used)))
        print('-' * 50)

        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
    else:
        # overwrite the file if recover_num is not set
        outfile = open(pathname, 'w')
        print('start from scratch')

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
        writer.writeheader()
    
    count = 0
    tp = 0
    query_len = 0
    for context_id, annotation in inf.items():
        context = annotation["passage"]

        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]

            question = qa_pair["question"]

            candidate_answers = [qa_pair["answer"]]
            if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                candidate_answers += qa_pair["validated_answers"]           

            # for answer in candidate_answers:
            #     gold_answer, gold_type = answer_json_to_strings(answer)

            #     if numerical_only and gold_type != "number":
            #         continue

            if count >= st + recover_num and count < en and query_id not in used:
                subqs, tmp_query_len = model(context, question, backbone)
                query_len += tmp_query_len

                #print('gpu' + str(gpu_id) + ': ' + str(count - st))
                #print(context)
                #print(question)
                print('-' * 10)
                print(subqs)
                print(tmp_query_len)
                print('=' * 50)

                writer.writerow({'query_id': query_id, 'subqs': subqs})
            count += 1
            if count >= st + recover_num:
                print('recover_num: ', count)
            if count >= en:
                print(f"GPU {gpu_id} finished!")
                outfile.close()
                return 

from collections import Counter

def most_common_string(lst):
    count = Counter(lst)
    most_common_element = count.most_common(1)
    if most_common_element:
        return most_common_element[0][0]
    else:
        return ''

def get_golden(whether_train):
    if whether_train:
        inf = json.load(open('../drop_dataset_with_test_questions/drop_dataset_train.json'))
    else:
        inf = json.load(open('../drop_dataset_with_test_questions/drop_dataset_dev.json'))
    golden_ans = {}
    major_ans = {}
    full_context = {}
    full_question = {}
    for context_id, annotation in inf.items():
        context = annotation["passage"]

        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            tmpans = []
            question = qa_pair["question"]

            candidate_answers = [qa_pair["answer"]]
            if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                candidate_answers += qa_pair["validated_answers"]           

            for answer in candidate_answers:
                gold_answer, gold_type = answer_json_to_strings(answer)
                tmpans.append(gold_answer[0].replace('-', ' ').lower())

            golden_ans[query_id] = tmpans
            major_ans[query_id] = most_common_string(tmpans)
            full_context[query_id] = context
            full_question[query_id] = question

    return golden_ans, major_ans, full_context, full_question

def clean_drop(preds):
    print('-'* 10)
    if 'The answer is' in preds:
        preds = preds.split('The answer is')[-1]
    if ': ' in preds:
        preds = preds.split(': ')[-1]
    if preds[-1] == '.':
        preds = preds[:-1]
    if len(preds.split('\n')) > 1:
        preds = preds.split('\n')[-1]
    preds = preds.replace('-', ' ').replace(',', '').lower()
    print('predict: ', preds)
    return preds

def run_drop(model, modelname, datasetname, seed, backbone, subset=-1, qdname='gpt-3.5-turbo', gpu_id=0, total_gpu=8):
    inf = json.load(open('../drop_dataset_with_test_questions/drop_dataset_dev.json'))
    total = 31974
    if subset != -1:
        total = subset

    each_gpu_num = total // total_gpu
    st = gpu_id * each_gpu_num
    en = (gpu_id + 1) * each_gpu_num
    if gpu_id + 1 == total_gpu:
        en = total

    if isinstance(backbone, str):
        backbone_name = backbone
    else:
        backbone_name = backbone[2]

    if subset != -1:
        pathname = backbone_name+'-results/'+qdname+'-'+modelname+'_drop_'+str(subset)+'_'+seed+'.csv'
    else:
        pathname = backbone_name+'-results/'+qdname+'-'+modelname+'_drop_'+seed+'.csv'

    recover_num = 0
    used = set()
    fieldnames = ['Question', 'Golden', 'predict', 'query_id']
    if os.path.exists(pathname):
        # if para --decompose is set and the file exits, then append to the end of file
        with open(pathname) as csvfile:
            cr = csv.DictReader(csvfile)

            for row in cr:
                #print(row)
                used.add(row['query_id'])
                recover_num += 1
        print('gpu' + str(gpu_id) + ' recover_num: ' + str(recover_num))
        print('-' * 50)
        print(recover_num)

        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
    else:
        # otherwise, overwrite the file directly
        outfile = open(pathname, 'w')

        print('start from scratch')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

    count = 0
    tp = 0
    rewards = []
    for context_id, annotation in inf.items():
        context = annotation["passage"].replace('</s>', '')

        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]

            question = qa_pair["question"].replace('</s>', '')

            candidate_answers = [qa_pair["answer"]]
            if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                candidate_answers += qa_pair["validated_answers"]           

            #if count >= st + recover_num and count < en:
            if query_id not in used:
                preds = clean_drop(model(context, question, backbone))

                tmpans = []
                for answer in candidate_answers:
                    gold_answer, gold_type = answer_json_to_strings(answer)
                    tmpans.append(gold_answer[0].replace('-', ' ').lower())
                major_ans = most_common_string(tmpans).replace(',', '')

                em_score, f1_score, predicted_bags, gold_bags = get_metrics([preds for _ in range(len(tmpans))], tmpans)
                rewards.append(f1_score)
                #if major_ans in preds:
                #    rewards.append(1)
                #else:
                #    rewards.append(0)
                writer.writerow({'Question': context + ' ' + question, 'Golden': major_ans, 'predict': preds, 'query_id': query_id})
                print('Golden: ', major_ans)
                print(rewards)
                print('=' * 30)

            count += 1
            if count >= st + recover_num:
                print('recover_num: ', count)
            if count == subset:
                print(f"{sum(rewards)} / {count-recover_num} = {sum(rewards) / len(rewards)}")
                break

    print(f"GPU {gpu_id} finished!")
    outfile.close()

# from metrics import get_metrics
from process_drop import get_metrics
def run_dropsanity4(model, modelname, datasetname, seed, backbone, subset=-1, qdname='gpt-3.5-turbo', batch=1, whether_train=False, gpu_id=0, total_gpu=1):
    if subset == -1:
        ph = 'full_'
    else:
        ph = str(subset) + '_'

    if whether_train:
        ph += 'train-'
        # total = 77409
    else:
        ph += 'dev-'
        # total = 9536

    if isinstance(backbone, str):
        backbone_name = backbone
    else:
        backbone_name = backbone[2]
    #res_path = "/mnt/task_wrapper/user_output/artifacts/" + backbone_name+'-results/'
    #if not os.path.exists(res_path):
    #    os.makedirs(res_path)

    #if subset != -1:
    #    pathname = backbone_name+'-results/'+ ph + qdname + '-' + modelname + '_cot_decompose_drop_' + seed + '.csv'
    #    #decomposor_path = os.path.join('drop_golden-decomposing-by-' + qdname, str(subset) + 'dev-cot_decompose_drop' + '_0-1.csv')
    #    decomposor_path = os.path.join('golden-decomposing-by-' + qdname, ph + 'cot_decompose_drop' + '_0-1_'+seed+'.csv')
    #else:
    pathname = backbone_name+'-results/'+ ph + qdname + '-' + modelname + '_cot_decompose_drop_' + seed + '.csv'
    decomposor_path = os.path.join('golden-decomposing-by-' + qdname, ph + 'cot_decompose_drop' + '_0-1_'+seed+'.csv')

    print(pathname)
    print(decomposor_path)
    print(os.path.exists(decomposor_path))
    assert os.path.exists(decomposor_path)
    
    tmp = set()
    with open(decomposor_path) as csvfile:
        cr = csv.DictReader(csvfile)
        for row in cr:
            if row['query_id'] in tmp:
                print('repeated query_id: ' + str(row['query_id']))
            tmp.add(row['query_id'])
    total = len(tmp)
    each_gpu_num = total // total_gpu
    st = gpu_id * each_gpu_num
    en = (gpu_id + 1) * each_gpu_num
    if gpu_id + 1 == total_gpu:
        en = total
    print(st, en)

    golden_ans, major_ans, full_context, full_question = get_golden(whether_train)

    fieldnames = ['Question', 'Golden', 'predict', 'query_id']
    recover_num = 0
    used = set()
    if os.path.exists(pathname):
        # if para --decompose is set and the file exits, then append to the end of file
        with open(pathname) as csvfile:
            cr = csv.DictReader(csvfile)

            for row in cr:
                #print(row)
                used.add(row['query_id'])
                recover_num += 1
        print('-' * 50)
        print('left: ' + str(en - st - recover_num))
        print('recover_num: ' + str(recover_num))
        print('unique query: ' + str(len(used)))
        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
    else:
        # otherwise, overwrite the file directly
        outfile = open(pathname, 'w')

        print('start from scratch')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
 
    count = 0
    tp = 0
    ques, anss, subqss = [], [], []
    rewards = []
    with open(decomposor_path) as csvfile:
        cr = csv.DictReader(csvfile)

        for row in cr:
            query_id, subqs = row['query_id'], row['subqs']
            if count >= st and count < en and query_id not in used:
                #org_q, golden_a, subqs = row['Question'], row['Golden'], row['subqs']
                if 'vicuna' in qdname:
                    subqs = vicuna_extract_subqs(subqs.split('ASSISTANT: ')[-1]).strip()
                else:
                    subqs = vicuna_extract_subqs(subqs).strip()
                #print(subqs)
                #print(full_question[query_id].replace('</s>', ''))
                preds = model(full_context[query_id], full_question[query_id].replace('</s>', ''), subqs, backbone)
                writer.writerow({'Question': full_question[query_id].replace('</s>', ''), 'Golden': major_ans[query_id].replace(',', ''), 'predict': preds, 'query_id': query_id})
                print('-'* 10)
                print('question: ', full_question[query_id].replace('</s>', ''))
                print('subqs: ', subqs)
                if 'The answer is' in preds:
                    preds = preds.split('The answer is')[-1]
                if ': ' in preds:
                    preds = preds.split(': ')[-1]
                if preds[-1] == '.':
                    preds = preds[:-1]
                if len(preds.split('\n')) > 1:
                    preds = preds.split('\n')[-1]
                preds = preds.replace('-', ' ').replace(',', '').lower()
                print('predict: ', preds)
                em_score, f1_score, predicted_bags, gold_bags = get_metrics([preds for _ in range(len(golden_ans[query_id]))], golden_ans[query_id])
                rewards.append(f1_score)

                #writer.writerow({'Question': full_question[query_id].replace('</s>', ''), 'Golden': major_ans[query_id].replace(',', ''), 'predict': preds, 'query_id': query_id})
                print('Golden: ', golden_ans[query_id])
                print(rewards)
                print('=' * 30)
                used.add(query_id)
                print('left: ', en - st - len(used))

            count += 1
            if count >= en:
                outfile.close()
                return
    #with open('result.txt', 'a') as outf:
    #    outf.write('golden-decomposing-by-' + qdname + 'cot_decompose_gsm8k' + f"_{subset}_{seed}: {tp} / {count-recover_num} = {tp / (count-recover_num)} \n")

def decompose_gsm8k(model, modelname, datasetname, seed, backbone, subset=-1, whether_train=False, gpu_id=0, total_gpu=1):
    if whether_train:
        inf = open('./datasets/gsm8k/train.jsonl')
        total = 7473
    else:
        inf = open('./datasets/gsm8k/test.jsonl')
        total = 1319
    if subset != -1:
        total = subset
    each_gpu_num = total // total_gpu
    st = gpu_id * each_gpu_num
    en = (gpu_id + 1) * each_gpu_num
    if gpu_id + 1 == total_gpu:
        en = total
    print(st, en)

    if isinstance(backbone, str):
        backbone_name = backbone
    else:
        backbone_name = backbone[2]

    if subset == -1:
        ph = 'full_'
    else:
        ph = str(subset) + '_'

    if not os.path.exists('golden-decomposing-by-' + backbone_name+'/'):
        os.mkdir('golden-decomposing-by-' + backbone_name+'/')

    if whether_train:
        ph += 'train-'
    else:
        ph += 'dev-'

    pathname = 'golden-decomposing-by-' + backbone_name+'/'+ph+modelname+'_'+datasetname+'_'+str(gpu_id)+'-'+str(total_gpu)+'_'+seed+'.csv'
    print(pathname)

    recover_num = 0
    used = set()
    fieldnames = ['Question', 'Golden', 'subqs']
    if os.path.exists(pathname):
        with open(pathname) as csvfile:
            cr = csv.DictReader(csvfile)

            for row in cr:
                used.add(row['Question'])
                recover_num += 1
        print('gpu' + str(gpu_id) + ' recover_num: ' + str(recover_num))
        print('left: ' + str(total - len(used)))
        print('-' * 50)

        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
    else:
        outfile = open(pathname, 'w')
        print('start from scratch')

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
        writer.writeheader()
    
    import json
    count = 0
    tp = 0

    query_len = 0
    for line in inf:
        data = json.loads(line)
        q, a = data['question'], data['answer'].split('####')[-1].replace(',', '').strip()
        if count >= st and count < en and q not in used:
            subqs, tmp_query_len = model(q, backbone)
            subqs = subqs.replace(',', '').strip()
            print('*' * 50)
            print(q)
            print('-' * 20)
            print(subqs)
            print(tmp_query_len)
            query_len += tmp_query_len

            writer.writerow({'Question': q, 'Golden': a, 'subqs': subqs})
        count += 1
        if count >= st:
            print('recover_num: ', count)
        if count >= en:
            print(f"GPU {gpu_id} finished!")
            break
    print(query_len)
    inf.close()
    outfile.close()

def batch_decompose_gsm8k(model, modelname, datasetname, seed, backbone, subset=-1, recover_num=-1, batch=1):
    if subset != -1:
        if isinstance(backbone, str):
            backbone_name = backbone
        else:
            backbone_name = backbone[2]
        pathname = 'golden-decomposing-by-' + backbone_name+'/'+modelname+'_'+datasetname+'_'+str(subset)+'_'+seed+'.csv'
    else:
        pathname = 'golden-decomposing-by-' + backbone+'/'+modelname+'_'+datasetname+seed+'.csv'

    fieldnames = ['Question', 'Golden', 'subqs']
    if os.path.exists(pathname):
        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
    else:
        outfile = open(pathname, 'w')
        print('start from scratch')

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
        writer.writeheader()
    
    import json
    count = 0
    tp = 0
    ques, anss = [], []
    with open('./datasets/gsm8k/test.jsonl') as inf:
        for line in inf:
            data = json.loads(line)
            if count >= recover_num:
                q, a = data['question'], data['answer'].split('####')[-1].replace(',', '').strip()
                ques.append(q)
                anss.append(a)
                if len(ques) == batch or count == subset:
                    subqs = model(ques, backbone).replace(',', '').strip()

                    for i in range(batch):
                        writer.writerow({'Question': ques[i], 'Golden': anss[i], 'subqs': subqs[i]})
                    ques = []
                    anss = []
            count += 1
            print(count)

            if count == subset:
                break
    outfile.close()

def extract_subqs(subqs):
    #print('originial subqs: ', subqs)
    final = []
    for line in subqs.strip().split('\n'):
        if line.strip().startswith('decomposed subquestion'):
            # print(len('decomposed subquestion'))
            # 22
            final.append(line.strip()[23:])
    if final == []:
        print(subqs)
        exit(0)
    for i in range(len(final)):
        tmp = final[i].split()
        tmp[0] = tmp[0][:-1] + '.'
        final[i] = ' '.join(tmp)
    #print('\n'.join(final) + '\n')
    #print('*' * 50)
    return '\n'.join(final) + '\n'

def extract_subqs4infer(subqs):
    #print('originial subqs: ', subqs)
    final = []
    for line in subqs.strip().split('\n'):
        if line.strip().lower().startswith('decomposed subquestion'):
            # print(len('decomposed subquestion'))
            # 22
            final.append(line.strip())
    if final == []:
        print(subqs)
        exit(0)
    #print('\n'.join(final) + '\n')
    return '\n'.join(final) + '\n'

def vicuna_extract_subqs(subqs):
    #print('originial subqs: ', subqs)
    final = []
    for line in subqs.strip().split('\n'):
        if '?' in line and 'main question' not in line.lower():
            final.append(line.strip())
    #print('\n'.join(final) + '\n')
    #print('-' * 50)
    return '\n'.join(final) + '\n'

def add_commas(number_string):
    number = int(number_string)  # convert string to integer
    return '{:,}'.format(number)  # convert integer back to string and add commas

def purify_preds(anss, preds):
    correct = 0
    purified_preds = []
    for idx in range(len(preds)):
        print('ORIGINAL pred: ', preds[idx])
        pred = preds[idx].lower().split('the answer is')[-1][1:]
        #print('ORIGINAL pred: ', pred)

        golden_a = anss[idx]

        if ',' not in golden_a:
            varied_golden_a = add_commas(golden_a)
        else:
            varied_golden_a = golden_a
            golden_a = golden_a.replace(',', '')
        judge = 0
        for i in pred.replace('$', '').split():
            if i != '' and i[-1] == '.':
                tmp = i[:-1]
            else:
                tmp = i

            if tmp != '' and (tmp[-1] == '\'' or tmp[-1] == '\"'):
                tmp = tmp[:-1]
            if tmp != '' and tmp[-1] == '%':
                tmp = tmp[:-1]

            
            st = tmp.find('.')
            if st != -1:
                transform = True
                for tmp_idx in range(st+1, len(tmp)):
                    if tmp[tmp_idx] != '0':
                        tranform = False
                        break
                if transform:
                    tmp = tmp[:st]

            if golden_a == tmp or varied_golden_a == tmp:
                judge = 1
                pred = tmp
                break
 
        print('golden: ', golden_a)
        print('pred: ', pred)
        if judge == 1:
            print('Correct')
        else:
            print('Wrong')
        print('*' * 50)

        correct += judge
        purified_preds.append(pred)
    return correct, purified_preds

def run_gsm8ksanity4(model, modelname, datasetname, seed, backbone, subset=-1, qdname='gpt-3.5-turbo', batch=1, whether_train=False, gpu_id=0, total_gpu=1):
    if subset == -1:
        ph = 'full_'
    else:
        ph = str(subset) + '_'

    if whether_train:
        inf = open('./datasets/gsm8k/train.jsonl')
        ph += 'train-'
    else:
        inf = open('./datasets/gsm8k/test.jsonl')
        ph += 'dev-'

    if isinstance(backbone, str):
        backbone_name = backbone
    else:
        backbone_name = backbone[2]

    if not os.path.exists('golden-decomposing-by-' + backbone_name+'/'):
        os.mkdir('golden-decomposing-by-' + backbone_name+'/')
    if not os.path.exists(backbone_name+'-results/'):
        os.mkdir(backbone_name+'-results/')

    pathname = backbone_name+'-results/'+ ph + qdname + '-' + modelname + '_cot_decompose_gsm8k_' + str(gpu_id) + '-' + str(total_gpu) + '_' + seed + '.csv'
    #decomposor_path = 'golden-decomposing-by-' + qdname + '/' + ph + modelname + '_' + datasetname + '_' + str(gpu_id)+'-'+str(total_gpu)+'_'+seed+'.csv'
    decomposor_path = 'golden-decomposing-by-' + qdname + '/' + 'full_dev-gsm_cot_decompose_gsm8k_' + str(gpu_id) + '-' + str(total_gpu) + '_0123' + '.csv'
    #decomposor_path = "golden-decomposing-by-gpt-3.5-turbo/full_train-gsm_cot_decompose_gsm8k_0-1_0123.csv"

    print(pathname)
    print(decomposor_path)
    print(os.path.exists(decomposor_path))

    tmp = {}
    for line in inf:
        data = json.loads(line)
        q, a = data['question'], data['answer'].split('####')[-1].replace(',', '').strip()
        if q in tmp:
            print('repeated query_id: ' + q)
        tmp[q] = a
    total = len(tmp)
    inf.close()
    if subset != -1:
        total = subset
    each_gpu_num = total // total_gpu
    st = gpu_id * each_gpu_num
    en = (gpu_id + 1) * each_gpu_num
    if gpu_id + 1 == total_gpu:
        en = total
    print(st, en)

    ################
    recover_num = 0
    used = set()
    fieldnames = ['Question', 'Golden', 'subqs', 'predict']
    if os.path.exists(pathname):
        with open(pathname) as csvfile:
            cr = csv.DictReader(csvfile)

            for row in cr:
                used.add(row['Question'])
                recover_num += 1
        print('gpu' + str(gpu_id) + ' recover_num: ' + str(recover_num))
        print('left: ' + str(total - len(used)))
        print('-' * 50)

        outfile = open(pathname, 'a')
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        print('continue from last file')
    else:
        outfile = open(pathname, 'w')
        print('start from scratch')

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    
        writer.writeheader()

    print(pathname)
    print(decomposor_path)
    print(os.path.exists(decomposor_path))

    count = 0
    tp = 0
    tt = 0
    if not os.path.exists(decomposor_path):
        for org_q, a in tmp.items():
            #data = json.loads(line)
            #org_q, a = data['question'], data['answer'].split('####')[-1].replace(',', '').strip()

            context = '.'.join(org_q.split('.')[:-1])+'.'
            q = org_q.split('.')[-1].strip()

            if 'If' in q and ',' in q:
                condition = ','.join(q.split(',')[:-1]) + ','
                q = q.split(',')[-1].strip()
                context += ' ' + condition
            if count >= recover_num and count < en and q not in used:
                pred = model(context, q, '', backbone).replace(',', '').strip()
                judge, purified_preds = purify_preds([a], [pred])
                tp += judge
                tt += 1
                print(f"{tp} / {tt} = {tp / (tt)}")
                print('recover_num: ', count+1)
                writer.writerow({'Question': org_q, 'Golden': a, 'subqs': ' ', 'predict': pred})
            count += 1
            if count >= st + recover_num:
                print('recover_num: ', count)
            if count >= en:
                print(f"GPU {gpu_id} finished!")
                break
        outfile.close()
        return 

    query_len = 0
    org_qs, anss, subqss = [], [], []
    with open(decomposor_path) as csvfile:
        cr = csv.DictReader(csvfile)

        for row in cr:
            org_q, golden_a, subqs = row['Question'], row['Golden'], row['subqs']
            if count >= st + recover_num and count < en and org_q not in used:
                print('=' * 50)
                print(subqs)
                print('-' * 20)
                if 'vicuna' in qdname:
                    subqs = vicuna_extract_subqs(subqs.split('ASSISTANT: ')[-1]).strip()
                else:
                    subqs = vicuna_extract_subqs(subqs).strip()

                org_qs.append(org_q)
                anss.append(golden_a)
                subqss.append(subqs)

                if len(org_qs) == batch or count == subset:
                    if batch != 1:
                        preds = model(org_qs, subqss, backbone)
                    else:
                        #preds = [model(ques[0], subqss[0], backbone)]
                        preds, tmp_query_len = model(org_qs[0], subqss[0], backbone)
                        preds = [preds]

                    judge, purified_preds = purify_preds(anss, preds)
                    tp += judge
                    tt += 1
                    query_len += tmp_query_len
                    print(f"{tp} / {tt} = {tp / (tt)}")
                    print('recover_num: ', count+1)
                    print(tmp_query_len)
                    for idx in range(len(purified_preds)):
                        writer.writerow({'Question': org_qs[idx], 'Golden': anss[idx], 'subqs': subqss[idx], 'predict': preds[idx]})
                    org_qs, anss, subqss = [], [], []

            count += 1
            if count >= en:
                print(f"GPU {gpu_id} finished!")
                break
    print(query_len)
    outfile.close()
def run_bamboogle(model, modelname, datasetname, seed, backbone):
    query_len, query_times = 0, 0
    with open(backbone+'-results/'+modelname+'_'+datasetname+seed+'.csv', 'w') as outfile:
        fieldnames = ['Question', 'Golden', 'predict']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        with open('./datasets/bamboogle.csv') as csvfile:
            cr = csv.DictReader(csvfile)
            for row in cr:
                pred, tmp_query_len, tmp_query_times = model(row['Question'], backbone) 
                query_len += tmp_query_len 
                query_times += tmp_query_times
                print(row['Question'])
                print('golden: ', row['Answer'])
                print('pred: ', pred)
                print('query_len: ', tmp_query_len)
                print('query_times: ', tmp_query_times)
                print('------------------')
                writer.writerow({'Question': row['Question'], 'Golden': row['Answer'], 'predict': pred})
    print(query_len)
    print(query_times)

def run_CC(model, modelname, datasetname, seed, backbone):
    import json

    count = 0
    with open('./tmpresults/'+modelname+'_'+datasetname+seed+'.csv', 'w') as outfile:
        fieldnames = ['Question', 'Golden', 'predict']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()

        with open('./datasets/compositional_celebrities_rest.csv') as csvfile:
            cr = csv.DictReader(csvfile)
            for row in cr:
                #count += 1
                #if count == 26:
                #    print(row['Question'])
                #if count > 26:
                pred = model(row['Question'], backbone) 
                print(str(count) + ' / 3139', row['Question'], row['Golden'], pred)
                writer.writerow({'Question': row['Question'], 'Golden': row['Golden'], 'predict': pred})                   

def load_vicuna_model(path, backbonename):
    print(path, backbonename)
    #path='lmsys/vicuna-13b-v1.3'
    whether_rl = False

    if 'rl' in path:
        path = os.path.join(path, 'best_checkpoint')
        whether_rl = True
    #path = '../realtest_1shot-qd_0shot_davinci003-ag/best_checkpoint'
    print("Loading model...", path)
    model = AutoModelForCausalLM.from_pretrained(path, low_cpu_mem_usage=True).cuda()
    print("Model loaded.")

    if not whether_rl:
        model.half().cuda()

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    tokenizer.padding_side = "left"
    print("Tokenizer loaded.")

    return (model, tokenizer, backbonename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default=None, help='the default backbone model to run')
    parser.add_argument('--backbone_path', type=str, default=None, help='the default backbone model to run')
    parser.add_argument('--qd', type=str, default='gpt-3.5-turbo', help='the question decomposor')
    parser.add_argument('--model', type=str, default=None, help='the default model to run')
    parser.add_argument('--dataset', type=str, default='bamboogle', help='the default dataset path')
    parser.add_argument('--subset', type=int, default=-1, help='the number downsampled examples.')
    parser.add_argument('--seed', type=str, default='1')
    parser.add_argument('--decompose', type=int, default=-1)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--whether_train', action="store_true", default=False)
    parser.add_argument('--numerical_only', action="store_true", default=False)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_gpu', type=int, default=8)
    args = parser.parse_args()

    print(args.model)
    model_dict = {'vinilla_ask': vinilla_ask, 'self_ask': self_ask, 'least_to_most_1_shot': least_to_most_1_shot, 'decompose': decompose, 'decompose2': decompose2, 'decompose3': decompose3, 'cot_1_shot': cot_1_shot, 'cot_4_shot': cot_4_shot, 'direct_decompose': direct_decompose, 'cot_decompose': cot_decompose, 'gsm_cot_decompose': gsm_cot_decompose, 'self_ask_decompose': self_ask_decompose, 'cot_qd': cot_qd, 'cot_infer_0': cot_infer_0, 'gsm_cot_infer_0': gsm_cot_infer_0, 'cot_infer_1': cot_infer_1, 'cot_infer_cot': cot_infer_cot, 'vinilla_vicuna': vinilla_vicuna, 'direct_qa': direct_qa, 'gsm_direct_qa': gsm_direct_qa, 'sft_qa': sft_qa}
    if args.model not in model_dict:
        exit(0)
    model = model_dict[args.model]


    if args.dataset == 'bamboogle':
        run_bamboogle(model, args.model, args.dataset, args.seed, args.backbone)

    if args.dataset == 'CC':
        run_CC(model, args.model, args.dataset, args.seed, args.backbone)

    if args.dataset == 'drop':
        if args.model == 'cot_decompose' or args.model == 'direct_decompose' or args.model == 'self_ask_decompose':
            # model should be either direct_decompose or cot_decompose
            if args.backbone == 'vinilla_vicuna':
                decompose_drop(model, args.model, args.dataset, args.seed, load_vicuna_model('lmsys/vicuna-13b-v1.3', args.backbone), args.decompose)
            elif '_qd-0shot_vicuna' in args.backbone or '_qd-1shot_vicuna' in args.backbone:
                decompose_drop(model, args.model, args.dataset, args.seed, load_vicuna_model(args.backbone_path, args.backbone), args.subset, args.whether_train, args.numerical_only, args.gpu_id, args.total_gpu)
            else:
                decompose_drop(model, args.model, args.dataset, args.seed, args.backbone, args.subset, args.whether_train, args.numerical_only, args.gpu_id, args.total_gpu)
        else:
            # decompose on test set
            if args.backbone == 'vinilla_vicuna':
                run_drop(model, args.model, args.dataset, args.seed, load_vicuna_model('lmsys/vicuna-13b-v1.3', args.backbone), args.subset, args.decompose)
            elif '_qd-0shot_vicuna' in args.backbone or '_qd-1shot_vicuna' in args.backbone:
                run_drop(model, args.model, args.dataset, args.seed, load_vicuna_model('../' + args.backbone + '-13b-v1.3', args.backbone), args.subset, args.decompose)
            else:
                run_drop(model, args.model, args.dataset, args.seed, args.backbone, args.subset, args.qd, args.gpu_id, args.total_gpu)

    if args.dataset == 'gsm8k':
        if args.model == 'gsm_cot_decompose' or args.model == 'direct_decompose' or args.model == 'self_ask_decompose':
            # model should be either direct_decompose or cot_decompose
            if args.backbone == 'vinilla_vicuna':
                decompose_gsm8k(model, args.model, args.dataset, args.seed, load_vicuna_model('lmsys/vicuna-13b-v1.3', args.backbone), args.subset, args.whether_train, args.gpu_id, args.total_gpu)
            elif '_qd-0shot_vicuna' in args.backbone or '_qd-1shot_vicuna' in args.backbone:
                decompose_gsm8k(model, args.model, args.dataset, args.seed, load_vicuna_model('../' + args.backbone + '-13b-v1.3/', args.backbone), args.subset, args.whether_train, args.gpu_id, args.total_gpu)
            else:
                decompose_gsm8k(model, args.model, args.dataset, args.seed, args.backbone, args.subset, args.whether_train, args.gpu_id, args.total_gpu)
        else:
            # decompose on test set
            if args.backbone == 'vinilla_vicuna':
                run_gsm8k(model, args.model, args.dataset, args.seed, load_vicuna_model('lmsys/vicuna-13b-v1.3', args.backbone), args.subset, args.decompose)
            elif '_qd-0shot_vicuna' in args.backbone or '_qd-1shot_vicuna' in args.backbone:
                run_gsm8k(model, args.model, args.dataset, args.seed, load_vicuna_model('../' + args.backbone + '-13b-v1.3', args.backbone), args.subset, args.decompose)
            else:
                run_gsm8k(model, args.model, args.dataset, args.seed, args.backbone, args.subset, args.decompose)

    if args.dataset == 'gsm8ksanity4':
        if args.backbone == 'vinilla_vicuna':
            run_gsm8ksanity4(model, args.model, args.dataset, args.seed, load_vicuna_model('lmsys/vicuna-13b-v1.3', args.backbone), args.subset, args.qd, args.batch, args.whether_train, args.gpu_id, args.total_gpu)
        elif '_qd-0shot_vicuna' in args.backbone or '_qd-1shot_vicuna' in args.backbone:
            run_gsm8ksanity4(model, args.model, args.dataset, args.seed, load_vicuna_model('../' + args.backbone + '-13b-v1.3/', args.backbone), args.subset, args.qd, args.batch, args.whether_train, args.gpu_id, args.total_gpu)
        else:
            run_gsm8ksanity4(model, args.model, args.dataset, args.seed, args.backbone, args.subset, args.qd, args.batch, args.whether_train)

    if args.dataset == 'dropsanity4':
        if args.backbone == 'vinilla_vicuna':
            run_dropsanity4(model, args.model, args.dataset, args.seed, load_vicuna_model('lmsys/vicuna-13b-v1.3', args.backbone), args.subset, args.qd, args.batch, args.whether_train, args.gpu_id, args.total_gpu)
        elif '_qd-0shot_vicuna' in args.backbone or '_qd-1shot_vicuna' in args.backbone:
            run_dropsanity4(model, args.model, args.dataset, args.seed, load_vicuna_model('../' + args.backbone + '-13b-v1.3/', args.backbone), args.subset, args.qd, args.batch, args.whether_train, args.gpu_id, args.total_gpu)
        else:
            run_dropsanity4(model, args.model, args.dataset, args.seed, args.backbone, args.subset, args.qd, args.batch, args.whether_train)

