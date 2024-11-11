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

    if len(after_colon) == 0:
        print('----empty case----')
        print(generated)
        return ''

    if '%' == after_colon[-1]:
        after_colon = after_colon[:-1]

    return after_colon

from metrics import get_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='the default model to run')
    parser.add_argument('--gpt', action='store_true')
    args = parser.parse_args()

    if os.path.exists(args.path):
        count, total = 0.0, 0.0
        b5 = 0.0
        l5 = 0.0
        f5 = 0.0
        e5 = 0.0
        out_tokens = 0
        if os.path.isdir(args.path):
            pths = [os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isfile(os.path.join(args.path, f))]
            print(pths)
        else:
            pths = [args.path]

        for pth in pths:
            with open(pth) as csvfile:
                cr = csv.DictReader(csvfile)
                for row in cr:
                    preds = row['predict']
                    if preds is None:
                        total += 1
                        continue
                    out_tokens += len(preds.split(' '))

                    if 'The answer is' in preds:
                        preds = preds.split('The answer is')[-1]
                    if ': ' in preds:
                        preds = preds.split(': ')[-1]
                    if preds != '' and preds[-1] == '.':
                        preds = preds[:-1]
                    if len(preds.split('\n')) > 1:
                        preds = preds.split('\n')[-1]
                    preds = preds.replace('-', ' ').replace(',', '').lower()
                    pred = extract_answer(preds.lower().strip())
                    l5 += len(pred)
                    if len(pred) > 5:
                        b5 += 1                   
                    em_score, f1_score, predicted_bags, gold_bags = get_metrics(pred, row['Golden'].lower().strip())
                    e5 += em_score
                    f5 += f1_score

                    if row['Golden'].lower().strip() in pred:
                        count += 1
                    #else:
                    #   print(row['Question'])
                    #   print(row['decomposition'])
                    #   print(row['Golden'].lower().strip())
                    #   print(pred)
                    #   print('------------')
                    #   print(row['predict'])
                    total += 1
                    #if total == 500:
                    #    print('Q:\t', row['Question'])
                    #    print('pred:\t', pred)
                    #    print('Golden:\t', row['Golden'])
                    #    break
        #total = 9536
        print('=' * 50)
        print('len: ', l5 / total)
        print('less < 5 percentage: ', 1 - b5 / total)
        print('f1: ', f5 / total)
        print('em: ', e5 / total)
        print(count / total)
        if not args.gpt:
            print('out vicuna price: ', out_tokens/1000 * 0.002 * 4/3 * 13/175)
        else:
            print('out price: ', out_tokens/1000 * 0.002 * 4/3)
        print(total)


