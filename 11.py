S = input()
i = 0
s = []
while i < len(S):
    if S[i] != " ":
        s.append(S[i])
        i = i + 1
    else:
        i = i + 1
print(''.join(s))
