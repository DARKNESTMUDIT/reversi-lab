"""The rules of 6×6 Reversi and its move vocabulary."""

from dataclasses import dataclass
from functools import cached_property

SIZE = 6
SQUARES = SIZE * SIZE
START = SQUARES
PASS = SQUARES + 1
END = SQUARES + 2
VOCAB = SQUARES + 3
BLACK, WHITE, EMPTY = 1, -1, 0

_OPENING = [EMPTY] * SQUARES
_OPENING[14] = _OPENING[21] = WHITE  # C3, D4
_OPENING[15] = _OPENING[20] = BLACK  # D3, C4
INITIAL_BOARD = tuple(_OPENING)


def _rays(square: int) -> tuple[tuple[int, ...], ...]:
    """Squares in each direction that could contain a bracketed disc."""
    row, column = divmod(square, SIZE)
    rays = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            path = []
            r, c = row + dr, column + dc
            while 0 <= r < SIZE and 0 <= c < SIZE:
                path.append(r * SIZE + c)
                r, c = r + dr, c + dc
            if len(path) >= 2:
                rays.append(tuple(path))
    return tuple(rays)


_RAYS = tuple(_rays(square) for square in range(SQUARES))


def _flips(board: tuple[int, ...], turn: int, square: int) -> tuple[int, ...]:
    opponent = -turn
    flipped = []
    for ray in _RAYS[square]:
        if board[ray[0]] != opponent:
            continue
        for distance in range(1, len(ray)):
            disc = board[ray[distance]]
            if disc == turn:
                flipped.extend(ray[:distance])
                break
            if disc == EMPTY:
                break
    return tuple(flipped)


@dataclass(frozen=True)
class State:
    """A board and the side to play, beginning with Black."""

    board: tuple[int, ...] = INITIAL_BOARD
    turn: int = BLACK
    ended: bool = False

    @cached_property
    def _placements(self) -> dict[int, tuple[int, ...]]:
        if self.ended:
            return {}
        return {
            square: flipped
            for square, disc in enumerate(self.board)
            if disc == EMPTY and (flipped := _flips(self.board, self.turn, square))
        }

    def flips(self, token: int) -> tuple[int, ...]:
        """Discs that a square placement would flip. Empty for other tokens."""
        return self._placements.get(token, ())

    def legal(self) -> tuple[int, ...]:
        if self._placements:
            return tuple(self._placements)
        if self.ended:
            return ()
        for square, disc in enumerate(self.board):
            if disc == EMPTY and _flips(self.board, -self.turn, square):
                return (PASS,)
        return (END,)

    def play(self, token: int) -> "State":
        if type(token) is not int or token not in self.legal():
            raise ValueError(f"Illegal move: {token!r}")
        if token == END:
            return State(self.board, self.turn, ended=True)
        if token == PASS:
            return State(self.board, -self.turn)
        board = list(self.board)
        for square in (token, *self._placements[token]):
            board[square] = self.turn
        return State(tuple(board), -self.turn)
