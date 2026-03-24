import patsy
import time
import sys
import pandas as pd
import numpy as np

import covbat as cb
from covbat_sklearn import CovBatHarmonizer

# import importlib
# importlib.reload(cb)

# read data from R output
pheno = pd.read_table('bladder-pheno.txt', index_col=0)
dat = pd.read_table('bladder-expr.txt', index_col=0)

mod = patsy.dmatrix("~ age + cancer", pheno, return_type="dataframe")

#### CovBat test ####
# record time
t = time.time()
ebat = cb.covbat(dat, pheno['batch'], mod, "age",pct_var=0.95, n_pc=0)

sys.stdout.write("covbat() took %.2f seconds\n" % (time.time() - t))

sys.stdout.write(str(ebat.iloc[:5, :5])+"\n")

#### CovBatHarmonizer sklearn test ####
# record time
t = time.time()
harmonizer = CovBatHarmonizer(pct_var=0.95, n_pc=0, numerical_covariates="age")
ebat_sklearn = harmonizer.fit_transform(dat, pheno['batch'], mod)

print(harmonizer)

sys.stdout.write("CovBatHarmonizer took %.2f seconds\n" % (time.time() - t))

sys.stdout.write(str(ebat_sklearn.iloc[:5, :5])+"\n")

assert (ebat - ebat_sklearn).max().max() < 1e-4

sys.stdout.write("PASS  test_covbat_sklearn\n")