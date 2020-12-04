import enum


class TrainMode(enum.Enum):
    FirstThenSecond = 0
    SecondThenFirst = 1
    EpochInterlaced = 2
    CycleInterlaced = 3
