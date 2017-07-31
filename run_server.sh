#!/bin/bash
# export PYTHONPATH="../:"$PYTHONPATH
export FLASK_DEBUG=1
export FLASK_APP=twapy/server.py
flask run
