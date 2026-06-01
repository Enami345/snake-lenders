"""Tiny gated print helper.

Gameplay prints (board generation, AI buy decisions, bankruptcies) are
useful when playing but flood the console during PPO training, where the
env steps thousands of times. Route those prints through `gprint` and let
training flip `VERBOSE` off.
"""

VERBOSE = True


def gprint(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)
"""(console_game UI prints stay on print() — they are the terminal UI.)"""
