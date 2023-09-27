import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"
# print(enc.encode("hello world"))
# print(enc.encode("hello"))
# print(enc.encode(" world"))

# print(enc.decode([15339, 1917]))
print(enc.encode("abab"))
print(enc.encode("ab"))

print(enc.encode("abcab"))
print(enc.encode("abzab"))
print(enc.encode("c"))
print(enc.encode("abc"))
print(enc.encode("政府"))
