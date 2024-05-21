# sample_size.py

import evoVAE.utils.statistics as stats
import pandas as pd
import numpy as np

DATA_PATH = "/scratch/user/s4646506/gfp_alns/independent_runs/"
ancestors: pd.DataFrame = pd.read_pickle(
    DATA_PATH + "GFP_AEQVI_full_04-29-2022_b08_ancestors_no_syn.pkl"
)

fracs = np.arange(0.01, 1.01, 0.01)


for frac in fracs:

    sub_sample = ancestors.sample(frac=frac, replace=False)

    mean_dev, std = stats.population_profile_deviation(ancestors, sub_sample)

    print(f"{frac},{mean_dev},{std},")

"""
import evoVAE.utils.seq_tools as st
import pandas as pd



DATA_PATH = "/scratch/user/s4646506/gfp_alns/independent_runs/"
ancestors: pd.DataFrame = pd.read_pickle(
    DATA_PATH + "GFP_AEQVI_full_04-29-2022_b08_ancestors_extants_no_syn.pkl"
)

fracs = np.arange(0.01, 1, 0.01)

for idx, frac in enumerate(fracs):

    sub_sample = ancestors.sample(frac=frac, replace=False)
    mean_dev = st.population_profile_deviation(ancestors, sub_sample)

    print(f"{frac},{mean_dev}")


import numpy as np
from scipy.stats import ttest_ind
from itertools import product

# Simulation 3: Sample size calculation for 80% power
for n in range(2, 101):
    sims = []

    for _ in range(10000):
        # Simulating data where the alternative hypothesis is true
        # and the true difference is 0.5
        placebo = np.random.normal(loc=0, scale=1, size=n)
        treatment = np.random.normal(loc=0.5, scale=1, size=n)

        # Run the hypothesis test
        _, p_value = ttest_ind(placebo, treatment)

        # Check if null hypothesis is rejected (p-value < 0.05)
        result = int(p_value < 0.05)

        sims.append(result)

    # Calculate the sample average of the simulations
    power = np.mean(sims)

    # Stop if we've achieved 80% power
    if power > 0.8:
        print(n)
        break

print(np.mean(sims))


"""
