# import の実験

import importlib

#mod = importlib.import_module('pickled_connection')  # OK
#mod = importlib.import_module('../python/pickled_connection')  # 失敗
mod = importlib.import_module('tmp.pickled_connection')  # OK

print(mod)
