#!/usr/bin/env python3
"""
Backtracking line search class plugin to be used with an L-BFGS optimization
https://en.wikipedia.org/wiki/Backtracking_line_search
"""
from seisflows import logger
from seisflows.plugins.line_search.bracket import Bracket
from seisflows.tools.math import parabolic_backtrack

class Nocedal(Bracket):
