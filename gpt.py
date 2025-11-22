import pygame
pygame.init()

screen = pygame.display.set_mode((1280,700))
clock = pygame.time.Clock()

sqaure_pos = pygame.Rect(650, 740, 50, 50)   # BAD spawn (but working for now)
circle_pos = pygame.Vector2(1100, 350)       # FIXED spawn (inside screen)

circle_spd = pygame.Vector2()
circle_rad = 20
circle_acc = 0.01
circle_spd_mul = 0.99
bounce_str = 1.0

while True:
    if pygame.event.get(pygame.QUIT):
        break

    keys = pygame.key.get_pressed()

    # Move square
    if keys[pygame.K_UP]:
        sqaure_pos.y -= 20
    if keys[pygame.K_DOWN]:
        sqaure_pos.y += 20
    if keys[pygame.K_LEFT]:
        sqaure_pos.x -= 20
    if keys[pygame.K_RIGHT]:
        sqaure_pos.x += 20

    # -------- SQUARE WALL LIMITS --------
    if sqaure_pos.x < 0:
        sqaure_pos.x = 0
    if sqaure_pos.x + sqaure_pos.width > screen.get_width():
        sqaure_pos.x = screen.get_width() - sqaure_pos.width
    if sqaure_pos.y < 0:
        sqaure_pos.y = 0
    if sqaure_pos.y + sqaure_pos.height > screen.get_height():
        sqaure_pos.y = screen.get_height() - sqaure_pos.height
    # -------------------------------------

    # BALL PHYSICS
    circle_spd *= circle_spd_mul
    cursor_pos = pygame.mouse.get_pos()
    circle_spd += (pygame.Vector2(cursor_pos) - circle_pos) * circle_acc

    # BALL WALL CHECKS
    if circle_pos.x < circle_rad:
        circle_pos.x = circle_rad
        circle_spd.x = -circle_spd.x
    if circle_pos.x > screen.get_width() - circle_rad:
        circle_pos.x = screen.get_width() - circle_rad
        circle_spd.x = -circle_spd.x

    if circle_pos.y < circle_rad:
        circle_pos.y = circle_rad
        circle_spd.y = -circle_spd.y
    if circle_pos.y > screen.get_height() - circle_rad:
        circle_pos.y = screen.get_height() - circle_rad
        circle_spd.y = -circle_spd.y

    # UPDATE BALL POSITION
    circle_pos += circle_spd

    # DRAW
    screen.fill("black")
    pygame.draw.circle(screen, "blue", circle_pos, circle_rad)
    pygame.draw.rect(screen, "red", sqaure_pos)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
