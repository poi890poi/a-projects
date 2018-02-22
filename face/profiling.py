import cProfile, re, pstats

import predict


cProfile.run('predict.main()', 'restats')
p = pstats.Stats('restats')
p.sort_stats('tottime')
p.print_stats(40)