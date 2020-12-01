import enum


class TrainMode(enum.Enum):
    FirstThenSecond = 0
    SecondThenFirst = 1
    Interlaced = 2
