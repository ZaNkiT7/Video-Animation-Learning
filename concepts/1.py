# a= 3+2
# print(a)

# a=3//3
# b="string"
# print(len(b+str(a))) # b+b,b+"ing",b+

# a="string"
# # b=a[0] # value of string
# for i in a: #this will consider i = val of a
#     print(i) # print whole string


a="aaaaa"
# while True:    #infinite loop
# while a.isdigit(): #false statement , a is not digit but string so no result
while not a.isprintable(): # a cant be printed    
    print(a.upper()) # in cases ,also dont forget()
else:
    print(a) # if still wanna run