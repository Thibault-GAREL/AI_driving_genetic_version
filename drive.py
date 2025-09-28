import pygame
import sys
import math

import ia

show = True
load = True

population_size = 100

WIDTH, HEIGHT = 1000, 600
# WIDTH, HEIGHT = 500, 800

if show:
    # Initialisation
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("IA de Course - Algorithme Génétique")

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GRAY = (40, 40, 60)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (255, 0, 255)

# Paramètres voiture
car_width, car_height = 40, 20
max_speed = 5
acceleration = 0.2
turn_speed = 4

# Murs
walls = [
    pygame.Rect(50, 50, 700, 10),
    pygame.Rect(50, 540, 700, 10),
    pygame.Rect(50, 50, 10, 500),
    pygame.Rect(740, 50, 10, 500),
    pygame.Rect(200, 150, 400, 10),
    pygame.Rect(200, 150, 10, 300),
    pygame.Rect(400, 250, 10, 300),
    pygame.Rect(600, 150, 10, 300),
]

# Checkpoints
checkpoints = [
    (180, 110, 40),  # Départ/arrivée
    (670, 150, 40),
    (600, 495, 40),
    (400, 210, 40),
    (180, 485, 40)
]


# # Paramètres voiture
# car_width, car_height = 40, 20
# max_speed = 5
# acceleration = 0.2
# turn_speed = 8
#
# # Murs
# walls = [
#     pygame.Rect(5, 5, 490, 10),
#     pygame.Rect(5, 750, 490, 10),
#     pygame.Rect(5, 5, 10, 750),
#     pygame.Rect(485, 5, 10, 750),
#     pygame.Rect(110, 150, 290, 10),
#     pygame.Rect(110, 150, 10, 480),
#     pygame.Rect(390, 150, 10, 480),
#     pygame.Rect(250, 300, 10, 450),
# ]
#
# # Checkpoints
# checkpoints = [
#     (100, 100, 40),  # Départ/arrivée
#     (400, 100, 40),
#     (400, 695, 40),
#     (250, 230, 40),
#     (100, 695, 40)
# ]


class Car:
    def __init__(self, brain=None):
        self.x = 100
        self.y = 100
        self.angle = 0
        self.speed = 0
        self.alive = True
        self.fitness = 0
        self.checkpoints_reached = 0
        self.current_checkpoint = 0
        self.distance_traveled = 0
        self.last_x, self.last_y = self.x, self.y
        self.stuck_timer = 0
        self.brain = ia.NeuralNetwork() if brain is None else brain

        # Performance tracking
        self.start_time = pygame.time.get_ticks()
        self.checkpoint_times = []

    def get_sensor_data(self):
        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        distances = []

        for sensor_angle in sensor_angles:
            distance = self.cast_ray(self.angle + sensor_angle)
            distances.append(distance)

        # Ajouter la vitesse comme entrée
        distances.append(self.speed / max_speed)

        return distances

    def cast_ray(self, angle):
        ray_x, ray_y = self.x, self.y
        step = 2
        max_distance = 200

        dx = math.cos(math.radians(angle)) * step
        dy = math.sin(math.radians(angle)) * step

        for i in range(int(max_distance / step)):
            ray_x += dx
            ray_y += dy

            if ray_x < 0 or ray_x >= WIDTH or ray_y < 0 or ray_y >= HEIGHT:
                return math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)

            for wall in walls:
                if wall.collidepoint(ray_x, ray_y):
                    return math.sqrt((ray_x - self.x) ** 2 + (ray_y - self.y) ** 2)

        return max_distance

    def update(self):
        if not self.alive:
            return

        # Obtenir les données des capteurs
        sensor_data = self.get_sensor_data()

        # Décision de l'IA
        outputs = self.brain.forward(sensor_data)

        # Interpréter les sorties
        accelerate = outputs[0] > 0.1
        brake = outputs[1] > 0.1
        turn_left = outputs[2] > 0.3
        turn_right = outputs[2] < -0.3

        # Mise à jour de la vitesse
        if accelerate and not brake:
            self.speed = min(self.speed + acceleration, max_speed)
        elif brake:
            self.speed = max(self.speed - acceleration, -max_speed / 2)
        else:
            self.speed *= 0.95

        # Rotation
        if turn_left:
            self.angle -= turn_speed
        elif turn_right:
            self.angle += turn_speed

        # Mouvement
        old_x, old_y = self.x, self.y
        dx = math.cos(math.radians(self.angle)) * self.speed
        dy = math.sin(math.radians(self.angle)) * self.speed
        new_x = self.x + dx
        new_y = self.y + dy

        # Vérifier les collisions
        if self.check_collision(new_x, new_y):
            self.alive = False
            return

        self.x, self.y = new_x, new_y

        # Calculer la distance parcourue
        self.distance_traveled += math.sqrt((self.x - old_x) ** 2 + (self.y - old_y) ** 2)

        # Vérifier les checkpoints
        self.check_checkpoints()

        # Détecter si la voiture est bloquée
        if abs(self.x - self.last_x) < 0.5 and abs(self.y - self.last_y) < 0.5:
            self.stuck_timer += 1
            if self.stuck_timer > 180:  # 3 secondes à 60 FPS
                self.alive = False
        else:
            self.stuck_timer = 0
            self.last_x, self.last_y = self.x, self.y

        # Calculer le fitness
        self.calculate_fitness()

    def check_collision(self, x, y):
        car_rect = pygame.Rect(0, 0, car_width, car_height)
        car_rect.center = (x, y)
        for wall in walls:
            if car_rect.colliderect(wall):
                return True
        return False

    def check_checkpoints(self):
        cx, cy, radius = checkpoints[self.current_checkpoint]
        distance = math.hypot(self.x - cx, self.y - cy)

        if distance < radius:
            if self.current_checkpoint == 0 and self.checkpoints_reached > 0:
                # Tour complet !
                current_time = pygame.time.get_ticks()
                lap_time = current_time - self.start_time
                self.checkpoint_times.append(lap_time)
                self.fitness += 10000  # Bonus énorme pour finir un tour

            self.checkpoints_reached += 1
            self.current_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
            self.fitness += 1000  # Bonus pour atteindre un checkpoint

    def calculate_fitness(self):
        # Fitness basé sur les checkpoints atteints et la distance
        checkpoint_bonus = self.checkpoints_reached * 1000
        distance_bonus = self.distance_traveled * 0.1

        # Bonus pour la progression vers le prochain checkpoint
        if self.current_checkpoint < len(checkpoints):
            cx, cy, _ = checkpoints[self.current_checkpoint]
            distance_to_checkpoint = math.hypot(self.x - cx, self.y - cy)
            proximity_bonus = max(0, 200 - distance_to_checkpoint)
        else:
            proximity_bonus = 0

        # Bonus de temps (survie)
        time_bonus = (pygame.time.get_ticks() - self.start_time) * 0.001

        self.fitness = checkpoint_bonus + distance_bonus + proximity_bonus + time_bonus

    def draw(self, color=RED):
        if not self.alive:
            color = (100, 100, 100)  # Gris pour les voitures mortes

        car_surface = pygame.Surface((car_width, car_height))
        car_surface.fill(color)
        car_surface.set_colorkey(BLACK)
        rotated_car = pygame.transform.rotate(car_surface, -self.angle)
        rect = rotated_car.get_rect(center=(self.x, self.y))
        screen.blit(rotated_car, rect.topleft)

    def draw_sensors(self):
        if not self.alive:
            return

        sensor_angles = [-45, -22.5, 0, 22.5, 45]
        for sensor_angle in sensor_angles:
            distance = self.cast_ray(self.angle + sensor_angle)
            end_x = self.x + math.cos(math.radians(self.angle + sensor_angle)) * distance
            end_y = self.y + math.sin(math.radians(self.angle + sensor_angle)) * distance
            pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)




def draw_checkpoints():
    font = pygame.font.Font(None, 24)
    for i, (x, y, radius) in enumerate(checkpoints):
        pygame.draw.circle(screen, BLUE, (x, y), radius, 3)
        label = "Départ" if i == 0 else f"{i}"
        text = font.render(label, True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        screen.blit(text, text_rect)


def game_loop():
    # print("Début game loop")
    if show:
        clock = pygame.time.Clock()
    ga = ia.GeneticAlgorithm(lambda brain=None: Car(brain), population_size)
    simulation_speed = 1
    if not show:
        simulation_speed = 1
    show_sensors = True

    if load:
        loaded_brain = ia.NeuralNetwork.load(ia.filename)
        car = Car(loaded_brain)  # dans drive.py

    if show:
        generation_timer = pygame.time.get_ticks()
        font = pygame.font.Font(None, 24)

    max_generation_time = 30000  # 30 secondes par génération
    max_generation_iteration = 5000

    iteration = 0

    done = False



    while not done:
        # print("Début loop")
        # print(f" i  = {iteration}")
        if show:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        simulation_speed = 3 if simulation_speed == 1 else 1
                    elif event.key == pygame.K_r:
                        ga.evolve()
                        generation_timer = pygame.time.get_ticks()
                    elif event.key == pygame.K_s:
                        show_sensors = not show_sensors

        # Mise à jour des voitures
        for _ in range(simulation_speed):
            alive_cars = 0
            for car in ga.cars:
                if car.alive:
                    car.update()
                    alive_cars += 1

        # Nouvelle génération si toutes les voitures sont mortes ou temps écoulé
        if show:
            current_time = pygame.time.get_ticks()
            if alive_cars <= population_size / 10 or (current_time - generation_timer) > max_generation_time:
                ga.evolve()
                generation_timer = current_time
        else:
            if alive_cars <= population_size / 10 or iteration > max_generation_iteration:
                ga.evolve()
                iteration = 0
                # generation_timer = current_time
        # Affichage
        if show:
            screen.fill(GRAY)

            # Dessiner le circuit
            for wall in walls:
                pygame.draw.rect(screen, WHITE, wall)

            draw_checkpoints()

        # Dessiner les voitures
        best_car = max(ga.cars, key=lambda car: car.fitness)
        if show:
            for i, car in enumerate(ga.cars):
                color = GREEN if car == best_car else RED

                car.draw(color)
                if show_sensors and car == best_car:
                    car.draw_sensors()


            # Interface
            gen_text = font.render(f"Génération: {ga.generation}", True, WHITE)
            fitness_text = font.render(f"Meilleur fitness: {ga.best_fitness:.1f}", True, WHITE)
            alive_text = font.render(f"Vivantes: {sum(1 for car in ga.cars if car.alive)}", True, WHITE)
            checkpoints_text = font.render(f"Checkpoints: {best_car.checkpoints_reached}", True, WHITE)

            controls_text = font.render("ESPACE: Accélérer | R: Nouvelle génération | S: Capteurs", True, WHITE)

            screen.blit(gen_text, (10, 10))
            screen.blit(fitness_text, (10, 35))
            screen.blit(alive_text, (10, 60))
            screen.blit(checkpoints_text, (10, 85))
            screen.blit(controls_text, (10, HEIGHT - 25))

            pygame.display.flip()
            clock.tick(60)
        iteration += 1

        # print("AAA")
        # print(f"i = {iteration}")

        # if iteration >= 30000:
        #     print("break")
        #     done = True
