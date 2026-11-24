"""Controlled bug reproduction: do not use .T on a four-dimensional key array.

This is a deliberately constructed debugging example, not a claim that the
teacher's starter contained an incorrect implementation of attention.
"""
import numpy as np

q = np.ones((2, 12, 50, 64), dtype=np.float32)
k = q.copy()
print('q:', q.shape, 'k:', k.shape, 'incorrect k.T:', k.T.shape)
try:
    scores = q @ k.T
except ValueError as error:
    print('Reproduced error:', type(error).__name__, str(error))
scores = q @ k.swapaxes(-1, -2)
print('Fixed key shape:', k.swapaxes(-1, -2).shape)
print('Fixed scores shape:', scores.shape, 'dtype:', scores.dtype)
