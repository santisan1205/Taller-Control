import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

# Definir la GPU dedicada
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Versión del algoritmo de entrenamiento (se guarda en config.json de cada corrida).
#   1: crítico con una sola salida, actualizado por turnos con la recompensa de cada
#      agente, y RND normalizando cada observación con su propia media y desviación.
#      Es con lo que se entrenaron la campaña v1 y la campaña v2 consolidada.
#   2: crítico con una salida por agente + normalización de valores (ValueNorm),
#      un solo optimizador del crítico, y RND con estadísticas acumuladas de las
#      observaciones y de la recompensa intrínseca.
VERSION_ALGORITMO = 2


class RunningMeanStd:
    """
    Media y varianza acumuladas (algoritmo paralelo de Chan et al.), por
    componente. Se usa para normalizar las observaciones y la recompensa
    intrínseca de RND, y los retornos del crítico (ValueNorm).
    """
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean, batch_var, batch_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m2 = self.var * self.count + batch_var * batch_count + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-8


class RNDModule(nn.Module):
    # Las observaciones normalizadas se recortan a [-CLIP_OBS, CLIP_OBS], como en el artículo de RND
    CLIP_OBS = 5.0

    def __init__(self, input_dim, output_dim=64, lr=1e-4):
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

        # Mediante descenso de gradiente se ajustan los pesos de la red predictora
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)

        # Estadísticas acumuladas: de cada componente de la observación (para
        # normalizarla) y de la recompensa intrínseca (para que su escala no
        # dependa de la inicialización de las redes ni decaiga a ~0)
        self.obs_rms = RunningMeanStd(shape=(input_dim,))
        self.r_int_rms = RunningMeanStd()
        self.ultima_r_int_cruda = 0.0

        # Enviar redes a la GPU
        self.target_net.to(device)
        self.predictor_net.to(device)

    def normalize_state(self, state):
        # Normaliza cada componente con la media y la desviación acumuladas de
        # las observaciones vistas (no con las del propio vector: eso borraba
        # la información de escala y hacía que estados distintos se vieran iguales)
        mean = torch.tensor(self.obs_rms.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.obs_rms.std, dtype=torch.float32, device=device)
        return torch.clamp((state - mean) / std, -self.CLIP_OBS, self.CLIP_OBS)

    def _a_tensor(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.asarray(state), dtype=torch.float32)
        return state.to(device)

    def compute_intrinsic_reward(self, state):
        """
        R_int = ||f(s') - g(s')||² / desviación acumulada de R_int. Actualiza las
        estadísticas de la observación con s' antes de normalizarla.
        """
        state = self._a_tensor(state)
        self.obs_rms.update(state.detach().cpu().numpy())
        with torch.no_grad():
            norm_state = self.normalize_state(state)
            target_features = self.target_net(norm_state)
            predicted_features = self.predictor_net(norm_state)
            # Error cuadrático medio entre las dos redes = novedad del estado
            r_int = float(((predicted_features - target_features) ** 2).mean(dim=-1))
        self.r_int_rms.update(r_int)
        self.ultima_r_int_cruda = r_int
        return r_int / float(self.r_int_rms.std)  # R_int normalizada

    def update_predictor(self, state):
        # Un paso de descenso de gradiente de la red predictora sobre s'
        state = self._a_tensor(state)
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
        # Pesos de importancia; w3 > w2 para priorizar el flujo peatonal.
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.p_cambio = p_cambio # Penalización leve por cambio de fase.

    def calculate_extrinsic_reward(self, delta_q, delta_w_veh, delta_w_ped, phase_changed):
        # Se emplea una Recompensa Basada en la Diferencia (Delta Reward) para garantizar estabilidad.
        penalizacion_cambio = self.p_cambio if phase_changed else 0.0

        # Fórmula: R_ext = -(w1 * delta_Q + w2 * delta_W_veh + w3 * delta_W_ped) - P_cambio.
        r_ext = -(self.w1 * delta_q + self.w2 * delta_w_veh + self.w3 * delta_w_ped) - penalizacion_cambio
        return r_ext

    def calculate_total_reward(self, r_ext, r_int, lambda_t):
        # El Actor-Crítico recibe la suma ponderada en el instante t.
        # Fórmula: R_total = R_ext + lambda_t * R_int.
        return r_ext + (lambda_t * r_int)


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

        # Desconectar log_prob de la GPU y del grafo de gradientes extrayendo solo su valor numérico
        return action.item(), log_prob.detach().cpu().item()


class MAPPOCritic(nn.Module):
    def __init__(self, global_obs_dim, n_agentes=1):
        """
        Crítico centralizado: observa el estado global de la red y estima el
        valor de cada agente (una salida por agente). Cada agente optimiza su
        propia recompensa, cuyas escalas son muy distintas (C2/C3 frente a C4),
        así que un único V(s) no puede ajustar los retornos de los cuatro.
        """
        super(MAPPOCritic, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(global_obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_agentes) # Salida: V_i(s) de cada agente (normalizado, ver CriticTrainer)
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
            torch.tensor(np.array(self.obs), dtype=torch.float32).to(device),
            torch.tensor(np.array(self.global_obs), dtype=torch.float32).to(device),
            torch.tensor(self.actions, dtype=torch.float32).to(device),
            torch.tensor(self.log_probs, dtype=torch.float32).to(device),
            torch.tensor(retornos, dtype=torch.float32).to(device),
            torch.tensor(ventajas, dtype=torch.float32).to(device)
        )


class MAPPOTrainer:
    def __init__(self, actor, lr_actor=3e-4, clip_ratio=0.2, entropy_coef=0.01):
        """
        Actualiza el Actor de un agente con el objetivo recortado de PPO. El
        Crítico compartido se actualiza aparte, una sola vez por episodio y con
        los retornos de todos los agentes (CriticTrainer).
        """
        self.actor = actor
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef # Fomenta la exploración

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

    def update(self, buffer_tensors, ppo_epochs=4):
        """
        Ejecuta la optimización utilizando un lote de trayectorias.
        """
        # Extraer tensores del buffer (calculados previamente con GAE)
        obs, _, old_actions, old_log_probs, _, advantages = buffer_tensors

        # Normalizar las ventajas estabiliza drásticamente el entrenamiento de la red
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Diagnósticos promediados sobre las épocas PPO (se registran en el log de entrenamiento)
        historial = {'actor_loss': [], 'entropia': [], 'kl_aprox': [], 'clip_frac': []}

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

            # 4. Backpropagation y Optimización
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # KL aproximada (pi_vieja || pi_nueva) y fracción de muestras recortadas por el clip:
            # indican qué tanto se mueve la política en cada actualización
            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs
                historial['kl_aprox'].append(((ratios - 1) - log_ratio).mean().item())
                historial['clip_frac'].append(((ratios - 1.0).abs() > self.clip_ratio).float().mean().item())
            historial['actor_loss'].append(actor_loss.item())
            historial['entropia'].append(entropy.item())

        return {clave: float(np.mean(valores)) for clave, valores in historial.items()}


class CriticTrainer:
    def __init__(self, critic, n_agentes, lr_critic=1e-3):
        """
        Entrena el Crítico centralizado con un solo optimizador. Usa ValueNorm
        (como MAPPO, Yu et al. 2022): la red predice retornos normalizados por
        agente con media y desviación acumuladas, y se desnormalizan al usarlos
        en GAE. Así las cuatro salidas tienen la misma escala aunque las
        recompensas de cada cruce difieran en órdenes de magnitud.
        """
        self.critic = critic
        self.norm = RunningMeanStd(shape=(n_agentes,))
        self.optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def valores(self, global_obs):
        """V_i(s) de cada agente, en la escala de las recompensas (numpy, una por agente)."""
        with torch.no_grad():
            normalizados = self.critic(torch.as_tensor(global_obs, dtype=torch.float32, device=device))
        return normalizados.cpu().numpy() * self.norm.std + self.norm.mean

    def update(self, global_obs, retornos, ppo_epochs=4):
        """
        global_obs: [T, dim_global]. retornos: [T, n_agentes] (columna i = retornos del agente i).
        Devuelve la pérdida media (MSE en unidades normalizadas).
        """
        self.norm.update(retornos.cpu().numpy())
        media = torch.tensor(self.norm.mean, dtype=torch.float32, device=device)
        std = torch.tensor(self.norm.std, dtype=torch.float32, device=device)
        objetivo = (retornos - media) / std
        perdidas = []
        for _ in range(ppo_epochs):
            perdida = nn.MSELoss()(self.critic(global_obs), objetivo)
            self.optimizer.zero_grad()
            perdida.backward()
            self.optimizer.step()
            perdidas.append(perdida.item())
        return float(np.mean(perdidas))
