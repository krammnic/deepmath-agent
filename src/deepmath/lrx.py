"""Executable reflected-witness construction from the 2026-09-25 LRX supplement.

Section 15 constructs a word with n(n-1)/2 letters.  Replaying this word verifies
its endpoint and gives an upper bound for that pair only.  The matching general
lower bound is a theorem reported by the source, not proved by this module.
"""

from __future__ import annotations


def _validate_size(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")
    if n < 4:
        raise ValueError("the reflected-witness theorem requires n >= 4")


def reflected_witness(n: int) -> tuple[int, ...]:
    """Return (1, 0, n-1, n-2, ..., 2), with rotations treated as paid moves."""
    _validate_size(n)
    return (1, 0, *range(n - 1, 1, -1))


def _endpoint_exchange(k: int) -> tuple[str, ...]:
    """E_0=X; E_(k+1)=XL E_k RX, exchanging a block's endpoints."""
    return ("X", "L") * k + ("X",) + ("R", "X") * k


def _prefix_reversal(j: int) -> tuple[str, ...]:
    """W_j reverses a length-j prefix and leaves floor(j/2)-1 left shifts."""
    word: list[str] = []
    while j >= 4:
        word.extend(_endpoint_exchange(j - 2))
        word.append("L")
        j -= 2
    word.extend(_endpoint_exchange(j - 2))
    return tuple(word)


def witness_word(n: int) -> tuple[str, ...]:
    """Return the Section 15 word taking identity to ``reflected_witness(n)``.

    Execute letters from left to right: L rotates left, R rotates right, and X
    swaps the first two entries.  Every letter, including the final RR, costs 1.
    The result has exactly n(n-1)/2 letters for every integer n >= 4.
    """
    _validate_size(n)
    a = n // 2
    b = n - a
    offset_a = a // 2 - 1
    offset_b = b // 2 - 1
    transfer = ("L" if offset_a == offset_b else "R",) * a
    inverse_letter = {"L": "R", "R": "L", "X": "X"}
    inverse_b = tuple(inverse_letter[letter] for letter in reversed(_prefix_reversal(b)))
    return _prefix_reversal(a) + transfer + inverse_b + ("R", "R")
