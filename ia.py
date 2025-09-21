import numpy as np

import random

filename = "best_brain__check4.npz"

class NeuralNetwork:
    def __init__(self, weights=None):
        # Réseau : 5 capteurs + vitesse -> 6 neurones cachés -> 3 sorties (accél, freiner, tourner)
        if weights is None:
            self.weights1 = np.random.randn(6, 8) * 2 - 1  # -1 à 1
            self.weights2 = np.random.randn(8, 3) * 2 - 1
        else:
            self.weights1, self.weights2 = weights

    def forward(self, inputs):
        # Normalisation des entrées
        inputs = np.array(inputs) / 200.0  # Normaliser les distances des capteurs
        inputs = np.clip(inputs, 0, 1)

        # Couche cachée
        hidden = np.tanh(np.dot(inputs, self.weights1))

        # Couche de sortie
        output = np.tanh(np.dot(hidden, self.weights2))

        return output

    def get_weights(self):
        return (self.weights1.copy(), self.weights2.copy())

    def save(self, filename):
        """Sauvegarde les poids du réseau dans un fichier .npz"""
        print(f"Save brain in {filename}")
        np.savez(filename, w1=self.weights1, w2=self.weights2)

    @classmethod
    def load(cls, filename):
        """Charge les poids depuis un fichier .npz et recrée un réseau"""
        data = np.load(filename)
        return cls((data["w1"], data["w2"]))

    def get_weights(self):
        return (self.weights1.copy(), self.weights2.copy())


class GeneticAlgorithm:
    def __init__(self, car_f, population_size=50):
        self.population_size = population_size
        self.generation = 1
        self.car_factory = car_f
        self.cars = [car_f() for _ in range(population_size)]
        self.best_fitness = 0
        self.best_car = None

    def selection(self):
        # Sélection par tournoi
        selected = []
        for _ in range(self.population_size):
            tournament = random.sample(self.cars, 5)
            winner = max(tournament, key=lambda car: car.fitness)
            selected.append(winner)
        return selected

    def crossover(self, parent1, parent2):
        # Croisement uniforme
        w1_p1, w2_p1 = parent1.brain.get_weights()
        w1_p2, w2_p2 = parent2.brain.get_weights()

        # Masque de croisement
        mask1 = np.random.random(w1_p1.shape) < 0.5
        mask2 = np.random.random(w2_p1.shape) < 0.5

        w1_child = np.where(mask1, w1_p1, w1_p2)
        w2_child = np.where(mask2, w2_p1, w2_p2)

        return NeuralNetwork((w1_child, w2_child))

    def mutate(self, brain, mutation_rate=0.3, mutation_strength=0.5): #0.1 / 0.5
        w1, w2 = brain.get_weights()

        # Mutation des poids
        if random.random() < mutation_rate:
            w1 += np.random.normal(0, mutation_strength, w1.shape)
        if random.random() < mutation_rate:
            w2 += np.random.normal(0, mutation_strength, w2.shape)

        return NeuralNetwork((w1, w2))

    def evolve(self):
        # Trier par fitness
        self.cars.sort(key=lambda car: car.fitness, reverse=True)

        # Garder les statistiques
        self.best_fitness = self.cars[0].fitness
        self.best_car = self.cars[0]

        print(f"Génération {self.generation}: Meilleur fitness = {self.best_fitness:.1f}, "
              f"Checkpoints = {self.best_car.checkpoints_reached}")

        if self.best_car.checkpoints_reached > 4:
            best_brain = self.best_car.brain
            best_brain.save(filename)

        # Nouvelle génération
        new_cars = []

        # Garder les 10% meilleurs (élitisme)
        elite_count = max(1, self.population_size // 10)
        for i in range(elite_count):
            new_cars.append(self.car_factory(self.cars[i].brain))

        # Créer le reste par croisement et mutation
        parents = self.selection()
        while len(new_cars) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child_brain = self.crossover(parent1, parent2)
            child_brain = self.mutate(child_brain)
            new_cars.append(self.car_factory(child_brain))

        self.cars = new_cars
        self.generation += 1

