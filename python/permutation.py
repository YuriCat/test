
def rotate(x, max_depth=1024):
    if max_depth == 0:
        return x
    if isinstance(x, (list, tuple)):
        if isinstance(x[0], (list, tuple)):
            return type(x[0])(
                rotate(type(x)(xx[i] for xx in x), max_depth - 1) \
                for i, _ in enumerate(x[0])
            )
        elif isinstance(x[0], dict):
            return type(x[0])(
                (key, rotate(type(x)(xx[key] for xx in x), max_depth - 1)) \
                for key in x[0]
            )
    elif isinstance(x, dict):
        x_front = x[list(x.keys())[0]]
        if isinstance(x_front, (list, tuple)):
            return type(x_front)(
                rotate(type(x)((key, xx[i]) for key, xx in x.items()), max_depth - 1) \
                for i, _ in enumerate(x_front)
            )
        elif isinstance(x_front, dict):
            return type(x_front)(
                (key2, rotate(type(x)((key1, xx[key2]) for key1, xx in x.items()), max_depth - 1)) \
                for key2 in x_front
            )
    return x


def permutate_impl(x, st):
    if len(st) == 0:
        return x

    if isinstance(x, (list, tuple)):
        y = type(x)(permutate_impl(xx, st[1:]) for xx in x)
    elif isinstance(x, dict):
        y = type(x)((key, permutate_impl(xx, st[1:])) for key, xx in x.items())

    return rotate(y, st[0])


def permutate(x, order):
    order_inv = [None for _ in order]
    for i, o in enumerate(order):
        order_inv[o] = i

    def strategy(oinv):
        if len(oinv) <= 1:
            return []

        s = strategy(oinv[1:])
        if len(s) > 0:
            oinv = [oinv[0]] + oinv[2: s[0] + 2] + [oinv[1]] + oinv[s[0] + 2:]

        for i, o in enumerate(reversed(oinv)):
            if oinv[0] >= o:
                break

        position = len(oinv) - 1 - i
        return [position] + s

    return permutate_impl(x, strategy(order_inv))


a = [(1, 2), [3, 4]]
print(a)
print(rotate(a))

a = {'a': [1, 2], 'b': [3, 4]}, {'a': [5, 6], 'b': [7, 8]}
print(a)
print(rotate(a))

print(rotate(a, 1))

print(permutate(a, [2, 0, 1]))

print(permutate(a, [1, 0]))