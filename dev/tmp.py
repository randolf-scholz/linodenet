from typing import Never


def foo() -> Never:
    while True:
        pass


def bar() -> tuple[int, ...]:
    return (foo(),)  # undetected unreachable code


x = bar()
x += (1, 2, 3)
