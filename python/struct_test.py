import struct

s = "3!i4"
n = struct.unpack("!i", s)

print(n)
