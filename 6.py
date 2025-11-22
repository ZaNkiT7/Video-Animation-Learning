import string
def fire(n,c=1):
    if c>len(n):
        return
    # print(''*(c-1)+'*'*(2*(n-c)+1)) # causing changes
    print([ch for ch in n])
    fire(n,c+1)
fire(string.digits)
fire(string.ascii_letters)
def nuke(n):
    a=[]
    for i in range(10):
        if (n>1):
            a.append(nuke(n-1))
        else:
            a.append(i)
    return a
print(nuke(10))