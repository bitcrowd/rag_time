#!/usr/bin/env python3

import os

from dotenv import dotenv_values


def load_env():
    attrs = dotenv_values()

    CODEBASE_PATH_ABS = os.path.abspath(attrs["CODEBASE_PATH"])
    VECTOR_DB_DIR_NAME = os.path.basename(CODEBASE_PATH_ABS) + "_vector_db"
    VECTOR_DB_PATH = os.path.join(
        os.path.dirname(CODEBASE_PATH_ABS), VECTOR_DB_DIR_NAME
    )

    attrs["CODEBASE_PATH"] = CODEBASE_PATH_ABS
    attrs["VECTOR_DB_PATH"] = VECTOR_DB_PATH
    attrs["CODE_SUFFIXES"] = list(map(str.strip, attrs["CODE_SUFFIXES"].split(",")))

    return attrs
