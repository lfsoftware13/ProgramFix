#!/usr/bin/env python
#
# Copyright 2007 Neal Norwitz
# Portions Copyright 2007 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenize C++ source code."""

import random
import sys

# Add $ as a valid identifier char since so much code uses it.
# C++0x string preffixes.
# Token types.

# Where the token originated from.  This can be used for backtracking.
# It is always set to WHENCE_STREAM in this code.



def fake_name(name):
    s = list(set(name))
    c = s[random.randint(0, len(s)-1)]
    a = random.randint(0, 1)
    INSERT = 0
    DELETE = 1
    if len(name) == 1:
        a = INSERT
    if a == INSERT:
        index = random.randint(0, len(name))
        return name[:index] + c +name[index:]
    elif a == DELETE:
        index = random.randint(0, len(name)-1)
        if index == 0:
            return name[1:]
        elif index == len(name) - 1:
            return name[:-1]
        else:
            return name[:index]+name[index+1:]
