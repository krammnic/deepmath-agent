"""Independent finite checks of the published constructive witness."""

from collections import deque
import unittest

from deepmath.lrx import reflected_witness, witness_word


def replay(state, letters):
    """Keep test semantics independent from production graph code."""
    state = list(state)
    for letter in letters:
        if letter == "L":
            state.append(state.pop(0))
        elif letter == "R":
            state.insert(0, state.pop())
        elif letter == "X":
            state[0], state[1] = state[1], state[0]
        else:
            raise AssertionError(f"invalid letter {letter!r}")
    return tuple(state)


def exact_distance(n, target):
    start = tuple(range(n))
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        state, distance = queue.popleft()
        if state == target:
            return distance
        for letter in "LRX":
            neighbor = replay(state, (letter,))
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, distance + 1))
    raise AssertionError("LRX target was unreachable")


class ReflectedWitnessTests(unittest.TestCase):
    def test_endpoints_and_exact_word_lengths(self):
        for n in range(4, 101):
            with self.subTest(n=n):
                word = witness_word(n)
                self.assertEqual(len(word), n * (n - 1) // 2)
                self.assertEqual(replay(range(n), word), reflected_witness(n))

    def test_small_exact_distances(self):
        # BFS proves only these finite instances; it is not the general theorem.
        for n in range(4, 8):
            with self.subTest(n=n):
                self.assertEqual(exact_distance(n, reflected_witness(n)), n * (n - 1) // 2)

    def test_relabeling_preserves_the_constructed_path(self):
        start = (20, 50, 10, 40, 30, 60, 0)
        expected = (start[1], start[0], *reversed(start[2:]))
        self.assertEqual(replay(start, witness_word(len(start))), expected)

    def test_size_domain_is_not_silently_extended(self):
        for function in (reflected_witness, witness_word):
            for n in (-1, 0, 1, 2, 3):
                with self.subTest(function=function.__name__, n=n):
                    with self.assertRaises(ValueError):
                        function(n)
            for n in (True, 4.0, "4", None):
                with self.subTest(function=function.__name__, n=n):
                    with self.assertRaises(TypeError):
                        function(n)


if __name__ == "__main__":
    unittest.main()
