#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import openai
import pandas as pd
import csv

import argparse

def extract_answer(generated):
    if '\n' not in generated:
        last_line =  generated
    else:
        last_line = generated.split('\n')[-1]

    if ':' not in last_line:
        after_colon = last_line.strip()
    else:
        after_colon = last_line.split(':')[-1].strip()

    if len(after_colon) == 0:
        print('----empty case----')
        print(generated)
        return ''

    if '.' == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon

def add_commas(number_string):
    number = int(number_string)  # convert string to integer
    return '{:,}'.format(number)  # convert integer back to string and add commas

def purify_preds(anss, pred):
    correct = 0
    golden_a = anss
    #pred = pred.lower().split('the answer is')[-1].replace(':', '')
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

        #if tmp != '' and (tmp[-1] == '\'' or tmp[-1] == '\"'):
        #    tmp = tmp[:-1]
        #if tmp != '' and tmp[-1] == '%':
        #    tmp = tmp[:-1]

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

    correct += judge
    return correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='the default model to run')
    parser.add_argument('--gpt', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.path):
        count, total = 0.0, 0.0
        out_tokens = 0
        count = []
        with open(args.path) as csvfile:
            cr = csv.DictReader(csvfile)
            for row in cr:
                if not args.gpt:
                    out_tokens += len(row['predict'].split(' '))
                else:
                    out_tokens += len(row['predict'].split('ASSISTANT:')[-1].split())
                preds = row['predict'].split('ASSISTANT:')[-1].strip()
                purified_preds = preds.split("The answer is: ")[-1].replace(',', '').replace('\"', '').lower()
                ans = row['Golden'].replace(',', '').replace('\"', '').lower()
                print(ans, purified_preds)

                count.append(ans in purified_preds)

                if ans not in purified_preds:
                    print(row['Question'])
                    print('golden:\t', row['Golden'])
                    print('predict:\t', row['predict'])
                    print('------------'*5)
                total += 1
                #if total == 200:
                #    break
        print(len(count))
        #print(count[-200:])
        #print(sum(count[-200:]) / 200)
        print(total)
        print('ACC:', sum(count) / total)

        print('=' * 50)
        if not args.gpt:
            print('out vicuna price: ', out_tokens/1000 * 0.002 * 4/3 * 13/175)
        else:
            print('out price: ', out_tokens/1000 * 0.002 * 4/3)

