import os
import pygame
from pong import Game
import neat
import pickle

WIDTH, HEIGHT = 800, 600

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball


    def test_ai(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            ai_output = net.activate((self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            ai_decision = ai_output.index(max(ai_output))

            if ai_decision == 0:
                pass
            elif ai_decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            self.game.loop()
            self.game.draw(True, False)
            pygame.display.update()

        pygame.quit()

    def train_ai(self, genome1, genome2, config):
        left_net = neat.nn.FeedForwardNetwork.create(genome1, config)
        right_net = neat.nn.FeedForwardNetwork.create(genome2, config)

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()

            left_output = left_net.activate((self.left_paddle.y , self.ball.y, abs(self.left_paddle.x - self.ball.x)))
            left_decision = left_output.index(max(left_output))

            if left_decision == 0:
                pass
            elif left_decision == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            right_output = right_net.activate((self.right_paddle.y , self.ball.y, abs(self.right_paddle.x - self.ball.x)))
            right_decision = right_output.index(max(right_output))

            if right_decision == 0:
                pass
            elif right_decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw(False, True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                genome1.fitness += game_info.left_hits
                genome2.fitness += game_info.right_hits
                break


def eval_genomes(genomes, config):
    window = pygame.display.set_mode((WIDTH, HEIGHT))

    for i, (genome_id, genome) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome.fitness = 0

        for genome_id_2, genome_2 in genomes[i+1:]:
            genome_2.fitness = 0 if genome_2.fitness == None else genome_2.fitness
            game = PongGame(window, WIDTH, HEIGHT)
            game.train_ai(genome, genome_2, config)


def run_neat(config):
    population = neat.Checkpointer.restore_checkpoint('neat-checkpoint-35')  # Read from checkpoint to avoid learning from 0
    # population = neat.Population(config)  # Uncomment if you want learn from start, comment line above

    stats = neat.StatisticsReporter()
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))  # Save checkpoint after every generations

    winner = population.run(eval_genomes, 50)

    with open('best_net.pickle', 'wb') as file:
        pickle.dump(winner, file)


def test_ai(config):

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    with open('best_net.pickle', 'rb') as file:
        winner = pickle.load(file)
    game = PongGame(window, WIDTH, HEIGHT)
    game.test_ai(winner, config)


if __name__ == '__main__':
    local_direction = os.path.dirname(__file__)
    config_path = os.path.join(local_direction, 'config.txt')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    # run_neat(config)  # Uncomment to learn AI
    test_ai(config)  # Uncomment if you want to play with ai and comment line above

