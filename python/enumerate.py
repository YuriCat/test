lst = [1, 2, 3, 4]

for i, v in enumerate(lst):
    print(i, v)
    if v == 2:
        lst.pop(i)

print(lst)
