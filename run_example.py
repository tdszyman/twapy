"""Example script illustrating how to use the twapy module to compute a temporal word analogy.

Assuming that there are files called '1987.bin' and '1997.bin' in the 'data/' directory,
this script will load those models and find the 1997 analogue of 'reagan' in 1987 (which should
be 'cilnton').
"""

import twapy

model_dir = './models'
collection = twapy.ModelCollection(model_dir)
analogy = twapy.Analogy('reagan', '1987', '1997', collection=collection)
print(analogy)
# This prints the following:
# 1987 : reagan :: 1997 : clinton
