import pygame
pygame.init()
screen = pygame.display.set_mode((1280,700))
clock= pygame.time.Clock()
sqaure_pos=pygame.Rect(650,600,50,50)
circle_pos=pygame.Vector2(1100,350)
circle_spd=pygame.Vector2()
circle_rad= 20
circle_acc=0.01
circle_spd_mul =0.99
bounce_str=1.0
while True:
    if pygame.event.get(pygame.QUIT):break
    keys=pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        sqaure_pos.y -=20
    if keys[pygame.K_DOWN]:
        sqaure_pos.y +=20
    if keys[pygame.K_LEFT]:
        sqaure_pos.x -=20    
    if keys[pygame.K_RIGHT]:
        sqaure_pos.x +=20
    if sqaure_pos.left < 0:
        sqaure_pos.left = 0
    if sqaure_pos.right > 1280:
        sqaure_pos.right = 1280
    if sqaure_pos.top < 0:
        sqaure_pos.top = 0
    if sqaure_pos.bottom > 700:
        sqaure_pos.bottom = 700

    circle_spd *=circle_spd_mul
    cursor_pos = pygame.mouse.get_pos()
    circle_spd += (pygame.Vector2(cursor_pos) - circle_pos) * circle_acc
   
    if circle_pos[0]<circle_rad:
        circle_pos.update(circle_spd[0]*bounce_str,circle_spd[1])
        circle_spd.update(-circle_spd[0]*bounce_str,circle_spd[1])
    
    elif circle_pos[0]>screen.get_width()-circle_rad:
        circle_pos.update(screen.get_width()-circle_rad,circle_pos[1])
        circle_spd.update(-circle_spd[0]*bounce_str,circle_spd[1])
    
    elif circle_pos[1]<circle_rad:
        circle_pos.update(circle_pos[0],circle_rad)
        circle_spd.update(circle_spd[0],-circle_spd[1]*bounce_str)
    
    elif circle_pos[1]>screen.get_height()-circle_rad:
        circle_pos.update(circle_pos[0], screen.get_height()-circle_rad)
        circle_spd.update(circle_spd[0],-circle_spd[1]*bounce_str)
    circle_pos=circle_pos+circle_spd
    screen.fill("black")
    pygame.draw.circle(screen,"blue",circle_pos,circle_rad )
    pygame.draw.rect(screen,"red",sqaure_pos)
    pygame.display.flip()
    clock.tick(60)
pygame.quit()