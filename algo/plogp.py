
from decimal import *

for a in ['1', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5']:
    alpha = Decimal(a)
    print('alpha =', alpha)
    for p in ['1', '1e-1', '1e-2', '1e-3', '1e-4', '1e-5', '1e-6', '1e-7']:
        prob = Decimal(p)
        print(-(prob ** alpha) * prob.ln()) 
