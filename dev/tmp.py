def xor(x: bool, y: bool) -> int:
    if x:
        if y:
            result = 0
        else:
            result = 1

    else:
        if y:
            result = 1
        else:
            result = 0

    return result
