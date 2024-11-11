#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None, help='the default model to run')
    args = parser.parse_args()

    if os.path.exists(args.path):
        count, total = 0.0, 0.0
        with open(args.path) as csvfile:
            cr = csv.DictReader(csvfile)
            for row in cr:
                if row['predict'] is not None and row['Golden'].lower().strip().replace(',', '') in row['predict'].lower().strip().replace(',', ''):
                    count += 1
                else:
                    #print(row['Question'])
                    #print(row['decomposition'])
                    print(row['Golden'])
                    #print(row['predict'])
                    if row['predict'] is not None:
                        print(extract_answer(row['predict'].lower().strip()))
                    print('------------')
                total += 1
        print(count / total)
        print(total)

