import string as ammo
gun = list(ammo.ascii_uppercase)
# while True:
#     if not gun:
#         gun=list(ammo.ascii_uppercase)
while gun:
    print(gun.pop())
