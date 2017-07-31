"""This script will download a sample of the trained models which can be used to illustrate the
functionality of the twapy module.
"""

import os
import requests

models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
for fn in ['1987.bin', '1997.bin']:
    url = 'https://s3.amazonaws.com/affrication-twapy/' + fn
    print("Downloading {:}".format(url))
    r = requests.get(url)
    if r.status_code == 200:
        print('Received {:} bytes.'.format(len(r.content)))
        with open(os.path.join(models_dir, fn), 'wb') as f:
            f.write(r.content)
    else:
        print('Error ({:}) getting {:}.'.format(r.status_code, url))
