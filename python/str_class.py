
class A(str):
    @classmethod
    def from_int(cls, n):
        return cls('k' + str(n))

print(A('k1'))
print(A.from_int(2))

print({A('k1'): True, A.from_int(1): False, A.from_int(2): True})
