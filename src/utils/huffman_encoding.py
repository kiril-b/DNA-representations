import heapq
from collections import Counter

from typing_extensions import Self


class Node:
    def __init__(self, left: Self, right: Self):
        self.left = left
        self.right = right

    def walk(self, code: dict[str, str], acc: str) -> None:
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class Leaf(Node):
    def __init__(self, char: str):
        self.char = char

    def walk(self, code: dict[str, str], acc: str) -> None:
        code[self.char] = acc or "0"

def huffman_code(string: str) -> dict[str, str]:
    h: list[tuple[int, int, Node]] = []

    for ch, freq in Counter(string).items():
        h.append((freq, len(h), Leaf(ch)))

    heapq.heapify(h)
    count = len(h)

    while len(h) > 1:
        freq1, _count1, left = heapq.heappop(h)
        freq2, _count2, right = heapq.heappop(h)
        heapq.heappush(h, (freq1 + freq2, count, Node(left, right)))
        count += 1

    code: dict[str, str] = {}
    if h:
        [(_freq, _count, root)] = h
        root.walk(code, "")
    return code

def encode(string: str, code: dict[str, str]) -> str:
    return "".join(code[ch] for ch in string)
