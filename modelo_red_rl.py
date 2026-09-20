import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class RNDModule(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super(RNDModule, self).__init__()
        
        # Red Objetivo (g): Posee pesos aleatorios congelados.
        self.target_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        for param in self.target_net.parameters():
            param.requires_grad = False # Pesos congelados.
            
        # Red Predictora (f): Se entrena para minimizar la función de pérdida frente a la red objetivo.
        self.predictor_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Mediante descenso de gradiente se ajustan los pesos de la red predictora[cite: 12].
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=1e-4)
        
    def normalize_state(self, state):
        # Es fundamental normalizar el vector de observación (media 0, varianza 1) antes de pasarlo por las redes[cite: 12].
        # Esto evita inestabilidad numérica generada al mezclar distintas escalas[cite: 12].
        mean = state.mean()
        std = state.std() + 1e-8
        return (state - mean) / std

    def compute_intrinsic_reward(self, state):
        # Convertir a tensor si no lo es
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        # 1. Normalización del vector de observación[cite: 12].
        norm_state = self.normalize_state(state)
        
        # 2. Paso por la Red Objetivo (g) y la Red Predictora (f)[cite: 12].
        target_features = self.target_net(norm_state)
        predicted_features = self.predictor_net(norm_state)
        
        # 3. Cálculo del Error Cuadrático Medio (MSE) para obtener la recompensa escalar[cite: 12].
        mse = nn.MSELoss(reduction='none')
        intrinsic_reward = mse(predicted_features, target_features).mean(dim=-1)
        
        return intrinsic_reward.detach().numpy() # R_int[cite: 12].
        
    def update_predictor(self, state):
        # Entrenamiento en lotes mediante retropropagación[cite: 12].
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
            
        norm_state = self.normalize_state(state)
        target_features = self.target_net(norm_state)
        predicted_features = self.predictor_net(norm_state)
        
        loss = nn.MSELoss()(predicted_features, target_features)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class RewardCalculator:
    def __init__(self, w1=1.0, w2=1.0, w3=2.0, p_cambio=0.5):
        # Pesos de importancia; w3 > w2 para priorizar el flujo peatonal[cite: 12].
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.p_cambio = p_cambio # Penalización leve por cambio de fase[cite: 12].
        
    def calculate_extrinsic_reward(self, delta_q, delta_w_veh, delta_w_ped, phase_changed):
        # Se emplea una Recompensa Basada en la Diferencia (Delta Reward) para garantizar estabilidad[cite: 12].
        penalizacion_cambio = self.p_cambio if phase_changed else 0.0
        
        # Fórmula: R_ext = -(w1 * delta_Q + w2 * delta_W_veh + w3 * delta_W_ped) - P_cambio[cite: 12].
        r_ext = -(self.w1 * delta_q + self.w2 * delta_w_veh + self.w3 * delta_w_ped) - penalizacion_cambio
        return r_ext

    def calculate_total_reward(self, r_ext, r_int, lambda_t):
        # El Actor-Crítico recibe la suma ponderada en el instante t[cite: 12].
        # Fórmula: R_total = R_ext + lambda_t * R_int[cite: 12].
        return r_ext + (lambda_t * r_int)


class TrafficLightAgent:
    def __init__(self, input_dim):
        # Espacio de acciones discreto (Discrete (4)) para Selección Directa de Fase[cite: 12].
        self.num_actions = 4
        self.rnd_module = RNDModule(input_dim=input_dim)
        self.reward_calc = RewardCalculator()
        
        # El coeficiente de exploración lambda_t sigue un decaimiento programado[cite: 12].
        self.lambda_t = 1.0 # Inicio del entrenamiento: lambda_t alto (ej. 1.0)[cite: 12].
        
    def decay_lambda(self, progress):
        # Decaimiento lineal progresivo. 
        # Mitad del entrenamiento: equivalente a 0.5[cite: 12].
        # Final del entrenamiento: cercano a 0 (ej. 0.01) para enfocarse 100% en optimizar el flujo vehicular[cite: 12].
        self.lambda_t = max(0.01, 1.0 - progress)
        
    def select_phase(self, action_index):
        # Las acciones se mapean a cadenas de estado predefinidas en SUMO[cite: 12].
        if action_index == 0:
            return "Fase 0 (Norte-Sur Directo + Peatones)" #[cite: 12].
        elif action_index == 1:
            return "Fase 1 (Norte-Sur Giros)" #[cite: 12].
        elif action_index == 2:
            return "Fase 2 (Este-Oeste Directo + Peatones)" #[cite: 12].
        elif action_index == 3:
            return "Fase 3 (Este-Oeste Giros)" #[cite: 12].
        
    def enforce_safety_constraints(self, current_phase, next_phase, green_time):
        # Restricciones de seguridad vial: Si cambia de fase, se fuerza fase amarilla y todo-rojo de despeje[cite: 12].
        if current_phase != next_phase:
            return "Iniciar transición (Amarillo -> Todo-Rojo)" #[cite: 12].
            
        # Garantizar tiempo mínimo (ej. 15s) para cruce peatonal seguro[cite: 12].
        if green_time < 15.0:
            return "Mantener fase (Tiempo verde mínimo no cumplido)" #[cite: 12].
            
        return "Cambio de fase permitido"
    
    
class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        """
        Actor descentralizado: Observa un cruce y decide la fase.
        """
        super(MAPPOActor, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim) # Salida: logits para cada fase (ej. 4 fases)
        )

    def forward(self, obs):
        # Convertir logits en probabilidades con Softmax
        logits = self.red(obs)
        probabilidades = torch.softmax(logits, dim=-1)
        
        # Crear una distribución categórica para muestrear la acción
        distribucion = Categorical(probabilidades)
        return distribucion

    def get_action(self, obs):
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob


class MAPPOCritic(nn.Module):
    def __init__(self, global_obs_dim):
        """
        Crítico centralizado: Observa toda la red para evaluar la acción conjunta.
        """
        super(MAPPOCritic, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Salida: Valor escalar del estado (V)
        )

    def forward(self, global_obs):
        valor_estado = self.red(global_obs)
        return valor_estado
    
    
class RolloutBuffer:
    def __init__(self):
        self.limpiar()

    def almacenar(self, obs, global_obs, action, log_prob, reward, value, done):
        """
        Guarda un paso de transición en la memoria.
        """
        self.obs.append(obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def limpiar(self):
        """
        Vacía el buffer después de cada actualización de la política.
        """
        self.obs = []
        self.global_obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def calcular_ventajas_gae(self, next_value, gamma=0.99, lam=0.95):
        """
        Calcula la Estimación de Ventaja Generalizada (GAE) matemática.
        Compara la recompensa real obtenida frente a la predicción del Crítico.
        """
        ventajas = []
        ventaja_acumulada = 0
        
        # Recorremos el buffer en reversa para propagar el valor futuro hacia atrás
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_val = self.values[t + 1]

            # Error TD (Diferencia Temporal)
            delta = self.rewards[t] + gamma * next_val * next_non_terminal - self.values[t]
            
            # GAE
            ventaja_acumulada = delta + gamma * lam * next_non_terminal * ventaja_acumulada
            ventajas.insert(0, ventaja_acumulada)
            
        retornos = [v + val for v, val in zip(ventajas, self.values)]
        
        # Convertir a tensores para el optimizador
        return (
            torch.tensor(self.obs, dtype=torch.float32),
            torch.tensor(self.global_obs, dtype=torch.float32),
            torch.tensor(self.actions, dtype=torch.float32),
            torch.tensor(self.log_probs, dtype=torch.float32),
            torch.tensor(retornos, dtype=torch.float32),
            torch.tensor(ventajas, dtype=torch.float32)
        )
        

class MAPPOTrainer:
    def __init__(self, actor, critic, lr_actor=3e-4, lr_critic=1e-3, clip_ratio=0.2, entropy_coef=0.01):
        """
        Gestiona la actualización de pesos de las redes Actor y Crítico.
        """
        self.actor = actor
        self.critic = critic
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef # Fomenta la exploración
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
    def update(self, buffer_tensors, ppo_epochs=4):
        """
        Ejecuta la optimización utilizando un lote de trayectorias.
        """
        # Extraer tensores del buffer (calculados previamente con GAE)
        obs, global_obs, old_actions, old_log_probs, returns, advantages = buffer_tensors
        
        # Normalizar las ventajas estabiliza drásticamente el entrenamiento de la red
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(ppo_epochs):
            # 1. Evaluar las acciones pasadas con la política ACTUAL
            dist = self.actor(obs)
            new_log_probs = dist.log_prob(old_actions)
            entropy = dist.entropy().mean()
            
            # 2. Ratio de probabilidad (pi_theta_nueva / pi_theta_vieja)
            ratios = torch.exp(new_log_probs - old_log_probs)
            
            # 3. Función de pérdida del Actor (Clipped Surrogate Objective)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            
            # Se minimiza el negativo de la ecuación para realizar ascenso de gradiente
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            
            # 4. Función de pérdida del Crítico (Mean Squared Error)
            state_values = self.critic(global_obs).squeeze()
            critic_loss = nn.MSELoss()(state_values, returns)
            
            # 5. Backpropagation y Optimización
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
        return actor_loss.item(), critic_loss.item()