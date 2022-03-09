import pygame


class Paddle:
    VEL = 5
    WIDTH, HEIGHT = 30, 100

    def __init__(self, x, y):
        self.x = self.starting_x = x
        self.y = self.starting_y = y


    def draw(self, win):
        pygame.draw.rect(win, (255, 255, 255), (self.x, self.y, self.WIDTH, self.HEIGHT))

    def move(self, up=True):
        if up:
            self.y -= self.VEL
        else:
            self.y += self.VEL

    def reset(self):
        self.x = self.starting_x
        self.y = self.starting_y

