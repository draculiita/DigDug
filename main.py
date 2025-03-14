import pygame
import sys
import os
import random
import math
from enum import Enum
import heapq
from collections import deque

# Inicialización de Pygame
pygame.init()
pygame.mixer.init()

# Constantes
TILE_SIZE = 32
GRID_WIDTH = 25
GRID_HEIGHT = 19
WIDTH = TILE_SIZE * GRID_WIDTH
HEIGHT = TILE_SIZE * GRID_HEIGHT
FPS = 60

# Colores
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 165, 0)

# Configuración de la pantalla
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dig Dug con IA")
clock = pygame.time.Clock()

# Carpetas para recursos
game_dir = os.path.dirname(__file__)
assets_dir = os.path.join(game_dir, "assets")
sound_dir = os.path.join(assets_dir, "sound")
img_dir = os.path.join(assets_dir, "img")

# Cargar sonidos y música
def load_sound(filename):
    return pygame.mixer.Sound(os.path.join(sound_dir, filename))

def load_music(filename):
    pygame.mixer.music.load(os.path.join(sound_dir, filename))

# Cargar imágenes
def load_image(filename, scale=1):
    img = pygame.image.load(os.path.join(img_dir, filename)).convert_alpha()
    if scale != 1:
        new_width = int(img.get_width() * scale)
        new_height = int(img.get_height() * scale)
        img = pygame.transform.scale(img, (new_width, new_height))
    return img

# Estados del juego
class GameState(Enum):
    MENU = 0
    GAME = 1
    GAME_OVER = 2
    WIN = 3

# Clase para el menú principal
class Menu:
    def __init__(self):
        self.title_font = pygame.font.Font(None, 64)
        self.option_font = pygame.font.Font(None, 36)
        self.title_text = self.title_font.render("DIG DUG", True, WHITE)
        self.start_text = self.option_font.render("Presiona ENTER para comenzar", True, WHITE)
        self.quit_text = self.option_font.render("Presiona ESC para salir", True, WHITE)
    
    def draw(self, surface):
        surface.fill(BLACK)
        title_rect = self.title_text.get_rect(center=(WIDTH // 2, HEIGHT // 3))
        start_rect = self.start_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        quit_rect = self.quit_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        
        surface.blit(self.title_text, title_rect)
        surface.blit(self.start_text, start_rect)
        surface.blit(self.quit_text, quit_rect)

# Clase para el grid del juego
class Grid:
    def __init__(self):
        self.width = GRID_WIDTH
        self.height = GRID_HEIGHT
        self.reset()
    
    def reset(self):
        # 0: Tierra (no excavada), 1: Túnel (excavado), 2: Roca
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Colocar algunas rocas aleatoriamente
        for _ in range(15):
            x, y = random.randint(1, self.width-2), random.randint(1, self.height-2)
            self.grid[y][x] = 2
    
    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_tunnel(self, x, y):
        if not self.is_valid_position(x, y):
            return False
        return self.grid[y][x] == 1
    
    def is_rock(self, x, y):
        if not self.is_valid_position(x, y):
            return False
        return self.grid[y][x] == 2
    
    def dig(self, x, y):
        if self.is_valid_position(x, y) and not self.is_rock(x, y):
            self.grid[y][x] = 1
            return True
        return False
    
    def draw(self, surface):
        for y in range(self.height):
            for x in range(self.width):
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                if self.grid[y][x] == 0:  # Tierra
                    pygame.draw.rect(surface, BROWN, rect)
                elif self.grid[y][x] == 1:  # Túnel
                    pygame.draw.rect(surface, LIGHT_BROWN, rect)
                elif self.grid[y][x] == 2:  # Roca
                    pygame.draw.rect(surface, (100, 100, 100), rect)
                
                # Dibujar líneas de la cuadrícula
                pygame.draw.rect(surface, (100, 50, 0), rect, 1)

# Implementación de A* para pathfinding
class AStar:
    def __init__(self, grid):
        self.grid = grid
    
    def heuristic(self, a, b):
        # Distancia Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        
        # Direcciones: arriba, derecha, abajo, izquierda
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.grid.is_valid_position(nx, ny) and not self.grid.is_rock(nx, ny):
                if self.grid.is_tunnel(nx, ny):
                    neighbors.append((nx, ny))
                # Si no es túnel, pero es tierra, considerarlo con mayor costo
                # (para que el enemigo prefiera seguir túneles existentes)
                elif self.grid.grid[ny][nx] == 0:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def find_path(self, start, goal):
        # Inicializar estructuras para A*
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        open_set_hash = {start}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)
            
            if current == goal:
                # Reconstruir el camino
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            for neighbor in self.get_neighbors(current):
                # Costo adicional si es necesario excavar
                dig_cost = 5 if not self.grid.is_tunnel(*neighbor) else 0
                tentative_g_score = g_score[current] + 1 + dig_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # No se encontró camino
        return None

# Nodos del Árbol de Comportamiento
class BTNode:
    def tick(self, actor):
        pass

class BTSelector(BTNode):
    def __init__(self, children):
        self.children = children
    
    def tick(self, actor):
        for child in self.children:
            if child.tick(actor):
                return True
        return False

class BTSequence(BTNode):
    def __init__(self, children):
        self.children = children
    
    def tick(self, actor):
        for child in self.children:
            if not child.tick(actor):
                return False
        return True

class BTCondition(BTNode):
    def __init__(self, condition_func):
        self.condition_func = condition_func
    
    def tick(self, actor):
        return self.condition_func(actor)

class BTAction(BTNode):
    def __init__(self, action_func):
        self.action_func = action_func
    
    def tick(self, actor):
        return self.action_func(actor)

# Clase del jugador
class Player(pygame.sprite.Sprite):
    def __init__(self, grid):
        pygame.sprite.Sprite.__init__(self)
        self.grid = grid
        # Posición inicial en el centro superior del mapa
        self.grid_x = GRID_WIDTH // 2
        self.grid_y = 1
        self.grid.dig(self.grid_x, self.grid_y)
        
        self.image = pygame.Surface((TILE_SIZE - 2, TILE_SIZE - 2))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.grid_x * TILE_SIZE + TILE_SIZE // 2
        self.rect.centery = self.grid_y * TILE_SIZE + TILE_SIZE // 2
        
        self.move_timer = 0
        self.move_delay = 150  # ms entre movimientos
        self.score = 0
        self.lives = 3
        self.pumping = False
        self.pump_target = None
        self.dig_sound = load_sound("dig.mp3")  # Asegúrate de tener este archivo
    
    def update(self, current_time):
        keys = pygame.key.get_pressed()
        
        # Control de movimiento con temporizador para no moverse demasiado rápido
        if current_time - self.move_timer > self.move_delay:
            new_x, new_y = self.grid_x, self.grid_y
            moved = False
            
            if keys[pygame.K_LEFT] and self.grid_x > 0:
                new_x -= 1
                moved = True
            elif keys[pygame.K_RIGHT] and self.grid_x < GRID_WIDTH - 1:
                new_x += 1
                moved = True
            elif keys[pygame.K_UP] and self.grid_y > 0:
                new_y -= 1
                moved = True
            elif keys[pygame.K_DOWN] and self.grid_y < GRID_HEIGHT - 1:
                new_y += 1
                moved = True
            
            # Si intentó moverse y no es una roca
            if moved and not self.grid.is_rock(new_x, new_y):
                # Excavar si es necesario
                if not self.grid.is_tunnel(new_x, new_y):
                    self.grid.dig(new_x, new_y)
                    self.score += 1
                    self.dig_sound.play()
                
                # Actualizar posición
                self.grid_x, self.grid_y = new_x, new_y
                self.rect.centerx = self.grid_x * TILE_SIZE + TILE_SIZE // 2
                self.rect.centery = self.grid_y * TILE_SIZE + TILE_SIZE // 2
                self.move_timer = current_time
    
    def pump(self, enemies):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_SPACE]:
            if not self.pumping:
                # Buscar enemigo cercano
                for enemy in enemies:
                    dx = abs(enemy.grid_x - self.grid_x)
                    dy = abs(enemy.grid_y - self.grid_y)
                    if dx <= 1 and dy <= 1:  # Enemigo adyacente
                        self.pumping = True
                        self.pump_target = enemy
                        break
        else:
            self.pumping = False
            self.pump_target = None
        
        # Si estamos bombeando un enemigo
        if self.pumping and self.pump_target:
            self.pump_target.get_pumped()
            # Sumar puntos por bombear
            self.score += 5
    
    def draw(self, surface):
        # En una versión completa, aquí cambiarías la imagen según la dirección, etc.
        pygame.draw.rect(surface, RED, self.rect)
        
        # Si está bombeando, dibujar la manguera
        if self.pumping and self.pump_target:
            pygame.draw.line(surface, WHITE, 
                             (self.rect.centerx, self.rect.centery),
                             (self.pump_target.rect.centerx, self.pump_target.rect.centery), 3)

# Clase base para enemigos
class Enemy(pygame.sprite.Sprite):
    def __init__(self, grid, player, x=None, y=None, color=ORANGE):
        pygame.sprite.Sprite.__init__(self)
        self.grid = grid
        self.player = player
        self.pathfinder = AStar(grid)
        
        # Posición inicial aleatoria (si no se especifica)
        if x is None or y is None:
            valid_pos = False
            while not valid_pos:
                self.grid_x = random.randint(1, GRID_WIDTH - 2)
                self.grid_y = random.randint(GRID_HEIGHT // 2, GRID_HEIGHT - 2)
                # Evitar posicionar sobre rocas o muy cerca del jugador
                valid_pos = (not self.grid.is_rock(self.grid_x, self.grid_y) and
                            (abs(self.grid_x - player.grid_x) > 5 or
                             abs(self.grid_y - player.grid_y) > 5))
        else:
            self.grid_x, self.grid_y = x, y
        
        self.grid.dig(self.grid_x, self.grid_y)  # Crear un túnel en la posición inicial
        
        self.image = pygame.Surface((TILE_SIZE - 4, TILE_SIZE - 4))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.centerx = self.grid_x * TILE_SIZE + TILE_SIZE // 2
        self.rect.centery = self.grid_y * TILE_SIZE + TILE_SIZE // 2
        
        self.move_timer = 0
        self.move_delay = 300  # ms entre movimientos (más lento que el jugador)
        self.path = []
        self.state = "patrol"
        self.patrol_points = []
        self.current_patrol_index = 0
        self.pump_count = 0
        self.max_pump = 3  # Cuántos bombeos aguanta antes de explotar
        
        # Crear puntos de patrulla aleatorios
        self.generate_patrol_points()
        
        # Comportamiento usando BT
        self.behavior_tree = self.create_behavior_tree()
    
    def generate_patrol_points(self):
        # Generar 3-5 puntos aleatorios para patrullar
        num_points = random.randint(3, 5)
        for _ in range(num_points):
            while True:
                px = random.randint(1, GRID_WIDTH - 2)
                py = random.randint(1, GRID_HEIGHT - 2)
                if not self.grid.is_rock(px, py):
                    self.patrol_points.append((px, py))
                    break
    
    def create_behavior_tree(self):
        # Crea y retorna el árbol de comportamiento específico para cada tipo de enemigo
        # Esta es una implementación básica que se sobrescribirá en las subclases
        return BTSelector([
            # Si está siendo bombeado, intentar huir
            BTSequence([
                BTCondition(lambda actor: actor.pump_count > 0),
                BTAction(lambda actor: actor.flee())
            ]),
            # Si el jugador está cerca, perseguirlo
            BTSequence([
                BTCondition(lambda actor: actor.is_player_nearby()),
                BTAction(lambda actor: actor.chase_player())
            ]),
            # De lo contrario, patrullar
            BTAction(lambda actor: actor.patrol())
        ])
    
    def is_player_nearby(self):
        # Considerar "cerca" si está a menos de 5 tiles
        dx = abs(self.player.grid_x - self.grid_x)
        dy = abs(self.player.grid_y - self.grid_y)
        return dx + dy < 5
    
    def chase_player(self):
        self.state = "chase"
        target = (self.player.grid_x, self.player.grid_y)
        self.path = self.pathfinder.find_path((self.grid_x, self.grid_y), target)
        return True
    
    def patrol(self):
        self.state = "patrol"
        if not self.patrol_points:
            self.generate_patrol_points()
            return False
        
        if not self.path or len(self.path) <= 1:
            # Ir al siguiente punto de patrulla
            target = self.patrol_points[self.current_patrol_index]
            self.path = self.pathfinder.find_path((self.grid_x, self.grid_y), target)
            
            if not self.path:  # Si no se puede llegar al punto, elegir otro
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
                return False
            
            # Si llegamos al punto de patrulla, ir al siguiente
            if len(self.path) <= 1:
                self.current_patrol_index = (self.current_patrol_index + 1) % len(self.patrol_points)
                return True
        
        return True
    
    def flee(self):
        # Intentar alejarse del jugador
        self.state = "flee"
        
        # Encontrar dirección opuesta al jugador
        dx = self.grid_x - self.player.grid_x
        dy = self.grid_y - self.player.grid_y
        
        # Normalizar y escalar para buscar un punto alejado
        dist = max(1, math.sqrt(dx*dx + dy*dy))
        dx = int(dx * 8 / dist)
        dy = int(dy * 8 / dist)
        
        target_x = min(max(1, self.grid_x + dx), GRID_WIDTH - 2)
        target_y = min(max(1, self.grid_y + dy), GRID_HEIGHT - 2)
        
        self.path = self.pathfinder.find_path((self.grid_x, self.grid_y), (target_x, target_y))
        
        # Si no se puede huir, moverse aleatoriamente
        if not self.path:
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = self.grid_x + dx, self.grid_y + dy
                if (self.grid.is_valid_position(nx, ny) and 
                    not self.grid.is_rock(nx, ny)):
                    self.grid_x, self.grid_y = nx, ny
                    self.rect.centerx = self.grid_x * TILE_SIZE + TILE_SIZE // 2
                    self.rect.centery = self.grid_y * TILE_SIZE + TILE_SIZE // 2
                    return True
        
        return bool(self.path)
    
    def get_pumped(self):
        self.pump_count += 1
        # Hinchar visualmente
        scale = 1 + (self.pump_count * 0.2)
        new_size = int((TILE_SIZE - 4) * scale)
        self.image = pygame.Surface((new_size, new_size))
        self.image.fill(self.image.get_at((0, 0)))  # Mantener el color original
        self.rect = self.image.get_rect(center=self.rect.center)
    
    def update(self, current_time):
        # Actualizar según el árbol de comportamiento
        self.behavior_tree.tick(self)
        
        # Explotar si está demasiado bombeado
        if self.pump_count >= self.max_pump:
            return True  # Indicar que debe ser eliminado
        
        # Movimiento según el camino calculado
        if current_time - self.move_timer > self.move_delay:
            if self.path and len(self.path) > 1:
                # Tomar el siguiente punto del camino
                next_pos = self.path[1]
                self.path = self.path[1:]
                
                # Excavar si es necesario
                if not self.grid.is_tunnel(*next_pos):
                    self.grid.dig(*next_pos)
                
                # Mover al enemigo
                self.grid_x, self.grid_y = next_pos
                self.rect.centerx = self.grid_x * TILE_SIZE + TILE_SIZE // 2
                self.rect.centery = self.grid_y * TILE_SIZE + TILE_SIZE // 2
                self.move_timer = current_time
            
            # Si está persiguiendo y perdimos el camino, recalcular
            elif self.state == "chase" or self.state == "flee":
                if self.state == "chase":
                    self.chase_player()
                else:
                    self.flee()
        
        return False  # No eliminar
    
    def draw(self, surface):
        pygame.draw.rect(surface, self.image.get_at((0, 0)), self.rect)
        
        # Dibujar ojos
        eye_radius = max(2, int(self.rect.width / 10))
        eye_offset = max(2, int(self.rect.width / 5))
        
        pygame.draw.circle(surface, WHITE, 
                          (self.rect.centerx - eye_offset, self.rect.centery - eye_offset), 
                          eye_radius)
        pygame.draw.circle(surface, WHITE, 
                          (self.rect.centerx + eye_offset, self.rect.centery - eye_offset), 
                          eye_radius)

# Subclases específicas de enemigos con comportamientos diferentes
class Pooka(Enemy):
    def __init__(self, grid, player, x=None, y=None):
        super().__init__(grid, player, x, y, ORANGE)
        self.move_delay = 250  # Ligeramente más rápido
    
    def create_behavior_tree(self):
        return BTSelector([
            # Si está siendo bombeado, intentar huir
            BTSequence([
                BTCondition(lambda actor: actor.pump_count > 0),
                BTAction(lambda actor: actor.flee())
            ]),
            # Si el jugador está muy cerca, perseguirlo agresivamente
            BTSequence([
                BTCondition(lambda actor: actor.is_player_very_close()),
                BTAction(lambda actor: actor.chase_player())
            ]),
            # Si el jugador está cerca, perseguirlo con cautela
            BTSequence([
                BTCondition(lambda actor: actor.is_player_nearby()),
                BTAction(lambda actor: actor.cautious_chase())
            ]),
            # De lo contrario, patrullar
            BTAction(lambda actor: actor.patrol())
        ])
    
    def is_player_very_close(self):
        dx = abs(self.player.grid_x - self.grid_x)
        dy = abs(self.player.grid_y - self.grid_y)
        return dx + dy < 3
    
    def cautious_chase(self):
        # Solo perseguir si hay un camino directo
        self.state = "cautious"
        target = (self.player.grid_x, self.player.grid_y)
        path = self.pathfinder.find_path((self.grid_x, self.grid_y), target)
        
        if path and len(path) < 8:  # No perseguir demasiado lejos
            self.path = path
            return True
        else:
            return self.patrol()

class Fygar(Enemy):
    def __init__(self, grid, player, x=None, y=None):
        super().__init__(grid, player, x, y, GREEN)
        self.move_delay = 300  # Más lento
        self.fire_ready = True
        self.fire_cooldown = 0
        self.fire_direction = None
    
    def create_behavior_tree(self):
        return BTSelector([
            # Si está siendo bombeado, intentar huir
            BTSequence([
                BTCondition(lambda actor: actor.pump_count > 0),
                BTAction(lambda actor: actor.flee())
            ]),
            # Si puede lanzar fuego, prepararse
            BTSequence([
                BTCondition(lambda actor: actor.is_player_in_line()),
                BTAction(lambda actor: actor.prepare_fire())
            ]),
            # Si el jugador está cerca, perseguirlo
            BTSequence([
                BTCondition(lambda actor: actor.is_player_nearby()),
                BTAction(lambda actor: actor.chase_player())
            ]),
            # De lo contrario, patrullar
            BTAction(lambda actor: actor.patrol())
        ])
    
    def is_player_in_line(self):
        # Comprobar si el jugador está en línea horizontal o vertical
        dx = abs(self.player.grid_x - self.grid_x)
        dy = abs(self.player.grid_y - self.grid_y)
        
        # Debe estar en línea y a una distancia razonable
        if (dx == 0 and 1 <= dy <= 4) or (dy == 0 and 1 <= dx <= 4):
            # Verificar que no hay rocas en el camino
            path_clear = True
            if dx == 0:  # Vertical
                step = 1 if self.player.grid_y > self.grid_y else -1
                for y in range(self.grid_y + step, self.player.grid_y + step, step):
                    if self.grid.is_rock(self.grid_x, y):
                        path_clear = False
                        break
            else:  # Horizontal
                step = 1 if self.player.grid_x > self.grid_x else -1
                for x in range(self.grid_x + step, self.player.grid_x + step, step):
                    if self.grid.is_rock(x, self.grid_y):
                        path_clear = False
                        break
            
            return path_clear and self.fire_ready
        
        return False
    
    def prepare_fire(self):
        self.state = "fire"
        
        # Determinar dirección del fuego
        if self.player.grid_x == self.grid_x:  # Vertical
            self.fire_direction = (0, 1) if self.player.grid_y > self.grid_y else (0, -1)
        else:  # Horizontal
            self.fire_direction = (1, 0) if self.player.grid_x > self.grid_x else (-1, 0)
        
        # Iniciar enfriamiento de la habilidad de fuego
        self.fire_ready = False
        self.fire_cooldown = pygame.time.get_ticks() + 3000  # 3 segundos de enfriamiento
        
        return True
    
    def update(self, current_time):
        result = super().update(current_time)
        
        # Actualizar enfriamiento del fuego
        if not self.fire_ready and current_time > self.fire_cooldown:
            self.fire_ready = True
        
        return result
    
    def draw(self, surface):
        super().draw(surface)
        
        # Si está preparando fuego, dibujar indicador
        if self.state == "fire" and self.fire_direction:
            fire_length = 3
            dx, dy = self.fire_direction
            start_x = self.rect.centerx
            start_y = self.rect.centery
            end_x = start_x + dx * TILE_SIZE * fire_length
            end_y = start_y + dy * TILE_SIZE * fire_length
            
            pygame.draw.line(surface, (255, 0, 0), (start_x, start_y), (end_x, end_y), 4)
            
            # Dibujar llamas en el extremo
            flame_rect = pygame.Rect(0, 0, TILE_SIZE // 2, TILE_SIZE // 2)
            flame_rect.center = (end_x, end_y)
            pygame.draw.rect(surface, (255, 165, 0), flame_rect)

# Clase principal del juego
class Game:
    def __init__(self):
        # Configuración inicial
        self.grid = Grid()
        self.menu = Menu()
        self.state = GameState.MENU
        self.player = None
        self.enemies = []
        self.last_enemy_spawn = 0
        self.enemy_spawn_delay = 10000  # 10 segundos entre enemigos
        self.game_over_sound = load_sound("game_over.mp3")
        self.win_sound = load_sound("win.mp3")
        self.background_music = load_music("background_music.mp3")
        
        # Iniciar música de fondo
        pygame.mixer.music.play(-1)
        
        # Fuentes para texto
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        
        # Objetivos del juego
        self.max_score = 500  # Puntuación para ganar
        self.max_enemies = 10  # Número máximo de enemigos a derrotar
        self.enemies_defeated = 0
    
    def reset_game(self):
        # Reiniciar el juego
        self.grid.reset()
        self.player = Player(self.grid)
        self.enemies = []
        self.last_enemy_spawn = pygame.time.get_ticks()
        self.enemies_defeated = 0
        
        # Añadir enemigos iniciales
        self.add_enemy()
        self.add_enemy()
    
    def add_enemy(self):
        # 70% probabilidad de Pooka, 30% de Fygar
        if len(self.enemies) < 5:  # Limitar número máximo de enemigos simultáneos
            if random.random() < 0.7:
                self.enemies.append(Pooka(self.grid, self.player))
            else:
                self.enemies.append(Fygar(self.grid, self.player))
    
    def handle_menu(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.state = GameState.GAME
                    self.reset_game()
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # Dibujar menú
        self.menu.draw(screen)
    
    def handle_game(self, events, current_time):
        # Procesar eventos
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.state = GameState.MENU
        
        # Actualizar jugador
        self.player.update(current_time)
        self.player.pump(self.enemies)
        
        # Actualizar enemigos y comprobar colisiones
        enemies_to_remove = []
        for i, enemy in enumerate(self.enemies):
            # Actualizar enemigo
            if enemy.update(current_time):
                enemies_to_remove.append(i)
                self.enemies_defeated += 1
                self.player.score += 100
            
            # Comprobar colisión con el jugador (si no está siendo bombeado)
            if (enemy.pump_count == 0 and 
                enemy.grid_x == self.player.grid_x and 
                enemy.grid_y == self.player.grid_y):
                self.player.lives -= 1
                if self.player.lives <= 0:
                    self.state = GameState.GAME_OVER
                    self.game_over_sound.play()
                else:
                    # Reposicionar al jugador
                    self.player.grid_x = GRID_WIDTH // 2
                    self.player.grid_y = 1
                    self.player.rect.centerx = self.player.grid_x * TILE_SIZE + TILE_SIZE // 2
                    self.player.rect.centery = self.player.grid_y * TILE_SIZE + TILE_SIZE // 2
        
        # Eliminar enemigos derrotados
        for i in sorted(enemies_to_remove, reverse=True):
            del self.enemies[i]
        
        # Generar nuevos enemigos
        if current_time - self.last_enemy_spawn > self.enemy_spawn_delay:
            self.add_enemy()
            self.last_enemy_spawn = current_time
        
        # Comprobar condiciones de victoria
        if self.player.score >= self.max_score or self.enemies_defeated >= self.max_enemies:
            self.state = GameState.WIN
            self.win_sound.play()
        
        # Dibujar
        self.draw_game()
    
    def handle_game_over(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.state = GameState.MENU
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # Dibujar pantalla de game over
        screen.fill(BLACK)
        game_over_text = self.big_font.render("GAME OVER", True, RED)
        score_text = self.font.render(f"Puntuación: {self.player.score}", True, WHITE)
        restart_text = self.font.render("Presiona ENTER para volver al menú", True, WHITE)
        
        screen.blit(game_over_text, game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 3)))
        screen.blit(score_text, score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        screen.blit(restart_text, restart_text.get_rect(center=(WIDTH // 2, HEIGHT * 2 // 3)))
    
    def handle_win(self, events):
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.state = GameState.MENU
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        
        # Dibujar pantalla de victoria
        screen.fill(BLACK)
        win_text = self.big_font.render("¡VICTORIA!", True, GREEN)
        score_text = self.font.render(f"Puntuación: {self.player.score}", True, WHITE)
        restart_text = self.font.render("Presiona ENTER para volver al menú", True, WHITE)
        
        screen.blit(win_text, win_text.get_rect(center=(WIDTH // 2, HEIGHT // 3)))
        screen.blit(score_text, score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        screen.blit(restart_text, restart_text.get_rect(center=(WIDTH // 2, HEIGHT * 2 // 3)))
    
    def draw_game(self):
        # Dibujar fondo y grid
        screen.fill(BLACK)
        self.grid.draw(screen)
        
        # Dibujar jugador y enemigos
        self.player.draw(screen)
        for enemy in self.enemies:
            enemy.draw(screen)
        
        # Dibujar HUD (información del jugador)
        score_text = self.font.render(f"Puntuación: {self.player.score}", True, WHITE)
        lives_text = self.font.render(f"Vidas: {self.player.lives}", True, WHITE)
        enemies_text = self.font.render(f"Enemigos: {self.enemies_defeated}/{self.max_enemies}", True, WHITE)
        
        screen.blit(score_text, (10, 10))
        screen.blit(lives_text, (10, 50))
        screen.blit(enemies_text, (WIDTH - 200, 10))
    
    def run(self):
        # Bucle principal del juego
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            events = pygame.event.get()
            
            for event in events:
                if event.type == pygame.QUIT:
                    running = False
            
            # Manejar estados del juego
            if self.state == GameState.MENU:
                self.handle_menu(events)
            elif self.state == GameState.GAME:
                self.handle_game(events, current_time)
            elif self.state == GameState.GAME_OVER:
                self.handle_game_over(events)
            elif self.state == GameState.WIN:
                self.handle_win(events)
            
            # Actualizar pantalla
            pygame.display.flip()
            clock.tick(FPS)
        
        pygame.quit()

# Iniciar el juego
if __name__ == "__main__":
    game = Game()
    game.run()
        
