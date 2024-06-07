#!/usr/bin/env python3

import os
from dotenv import dotenv_values

def load_env():
    attrs = dotenv_values()

    COCEBASE_PATH_ABS = os.path.abspath(attrs['COCEBASE_PATH'])
    VECTOR_DB_DIR_NAME = os.path.basename(COCEBASE_PATH_ABS) + "_vector_db"
    VECTOR_DB_PATH = os.path.join(os.path.dirname(COCEBASE_PATH_ABS), VECTOR_DB_DIR_NAME)

    attrs['CODEBASE_PATH'] = COCEBASE_PATH_ABS
    attrs['VECTOR_DB_PATH'] = VECTOR_DB_PATH

    return attrs