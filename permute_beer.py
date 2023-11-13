#!/usr/bin/env python3
""" Permute beer script
"""

import numpy as np

rng = np.random.default_rng()

import pandas as pd
pd.set_option('mode.copy_on_write', True)


df = pd.read_csv('data/mosquito_beer.csv')

after = df[df['test'] == 'after']
n_subs = len(after)

to_test = after[['group', 'activated']]
groups = np.array(to_test['group'])  # An array
activated = to_test['activated']  # A Series

group_means = activated.groupby(groups).mean()
actual_diff = np.diff(group_means)[0]

print('Actual diff', actual_diff)

n_iters = 10_000
fake_diffs = np.zeros(n_iters)

for i in range(n_iters):
    shuffled_groups = rng.permuted(groups)
    group_means = activated.groupby(shuffled_groups).mean()
    fake_diffs[i] = np.diff(group_means)[0]

p = np.count_nonzero(fake_diffs <= actual_diff) / n_iters

print('p', p)

import scipy.stats as sps

res = sps.ttest_ind(activated[groups == 'beer'],
                    activated[groups == 'water'],
                    permutations=10_000,
                    alternative='greater')

print('Scipy result', res)
