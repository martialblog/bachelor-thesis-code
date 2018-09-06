#!/usr/bin/env python3


import evaluate

standard_file = 'source/verb_tokens_test_gold_labels.csv'
all_zero = 'source/verb_tokens_all_zero.csv'
all_one = 'source/verb_tokens_all_one.csv'

all_std_results = evaluate.precision_recall_f1(standard_file, standard_file)
print('Gold Baseline')
print(all_std_results)

all_zero_results = evaluate.precision_recall_f1(all_zero, standard_file)
print('All Zero Baseline')
print(all_zero_results)

all_one_results = evaluate.precision_recall_f1(all_one, standard_file)
print('All One Baseline')
print(all_one_results)
