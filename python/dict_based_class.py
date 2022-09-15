class Dict(dict):
    def change(self, n):
        self['a']['b'] = n

    def new_change(self, n):
        a = Dict({**self, 'a': {'b': 1}})
        return a

a = Dict({'a': {'b': 0}})
b = Dict(a)
c = a.new_change(1)
print(a)
print(b)
print(c)
b.change(2)
a['a'] = None
print(a)
print(b)
print(c)
