
# python3 warning_test.py ... warning only
# python3 -W error warning_test.py ... exit after warning

import warnings

#warnings.simplefilter('error', RuntimeWarning)

for i in range(10):
    print(i)
    if i == 5:
        warnings.warn("bad", RuntimeWarning)

