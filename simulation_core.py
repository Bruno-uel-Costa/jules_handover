import random # Adicionado para BPP
import math # Adicionado para expovariate e log (para geométrica customizada)

# Placeholder for simulation core logic
# This file will contain the main simulation functions and execution block.

# --- Funções de Geração para BPP ---

def custom_geometric(p: float) -> int:
    """
    Gera um número aleatório seguindo uma distribuição geométrica com probabilidade de sucesso p.
    Retorna o número de tentativas até o primeiro sucesso. Mínimo 1.
    Implementação manual caso random.geometric não esteja disponível.
    """
    if not (0 < p <= 1):
        raise ValueError("A probabilidade 'p' deve estar em (0, 1].")
    # U ~ Uniform(0,1)
    # k = floor(log(U) / log(1-p)) + 1
    # No entanto, para evitar log(0) se random.random() retornar 0.0 (improvável mas possível),
    # usamos 1.0 - random.random() que está em (0, 1].
    # Ou, mais simples, random.random() já está em [0.0, 1.0), então 1.0 - random.random() está em (0.0, 1.0]
    # Se p=1, log(1-p) = log(0) -> erro. Se p=1, deve retornar 1 sempre.
    if p == 1.0:
        return 1

    u = random.random() # Gera um float no intervalo [0.0, 1.0)
    if u == 0.0: # Para evitar math.log(0) embora raro
        # Tratar como se fosse um número muito pequeno, resultando em k grande.
        # Ou, mais pragmaticamente, re-rolar ou retornar um valor grande.
        # Por simplicidade, vamos usar 1.0 - random.random() para garantir (0,1]
        # No entanto, a fórmula padrão é log(u), e u pode ser 0.
        # A biblioteca padrão do Python `random.geometric` lida com isso.
        # Uma abordagem comum é usar u em (0,1).
        # Se u for 0, log(u) é -inf.
        # Vamos usar um u no intervalo (0,1) explicitamente.
        u_exclusive_zero = 0.0
        while u_exclusive_zero == 0.0:
            u_exclusive_zero = random.random()
        return math.floor(math.log(u_exclusive_zero) / math.log(1.0 - p)) + 1

    return math.floor(math.log(u) / math.log(1.0 - p)) + 1


def generate_service_time(sim_params: dict) -> float:
    """
    Gera o tempo de serviço para uma chamada, assumindo distribuição exponencial.
    """
    mu = sim_params.get('mu')
    if mu is None or mu <= 0:
        raise ValueError("Taxa de serviço (mu) inválida ou ausente em sim_params.")
    return random.expovariate(mu)


def generate_batch_arrival_time(lambda_batch_rate: float) -> float:
    """
    Gera o tempo até a chegada do próximo lote de chamadas.
    Funcionalmente idêntico a gerar tempo de inter-chegada para um processo de Poisson.
    """
    if lambda_batch_rate <= 0:
        raise ValueError("A taxa de chegada do lote (lambda_batch_rate) deve ser positiva.")
    return random.expovariate(lambda_batch_rate)

def generate_batch_size(batch_params: dict) -> int:
    """
    Gera o número de chamadas em um lote com base nos parâmetros fornecidos.
    Atualmente, suporta apenas a distribuição geométrica.
    """
    if not isinstance(batch_params, dict):
        raise ValueError("batch_params deve ser um dicionário.")

    dist_type = batch_params.get('dist')
    if dist_type == 'geom':
        p = batch_params.get('p')
        if p is None or not (0 < p <= 1):
            raise ValueError("Parâmetro 'p' para distribuição geométrica é inválido ou ausente.")
        # random.geometric(p) retorna o número de tentativas até o primeiro sucesso.
        # Se o tamanho do lote deve ser no mínimo 1, isso já está correto.
        # size = random.geometric(p) # Substituído devido a AttributeError no ambiente
        size = custom_geometric(p)
        return size
    else:
        raise NotImplementedError(f"Distribuição de tamanho de lote '{dist_type}' não implementada.")

import heapq # Para FutureEventList

# --- Estruturas de Dados da Simulação ---

class SimEvent:
    def __init__(self, event_time: float, event_type: str, event_data: dict, priority: int = 0):
        self.event_time = event_time
        self.event_type = event_type
        self.event_data = event_data
        self.priority = priority # Para desempatar eventos no mesmo tempo, se necessário

    def __lt__(self, other):
        # Min-heap, então compara tempos de evento, depois prioridade
        if self.event_time == other.event_time:
            return self.priority < other.priority
        return self.event_time < other.event_time

class FutureEventList:
    def __init__(self):
        self._events = []

    def add_event(self, event: SimEvent):
        heapq.heappush(self._events, event)

    def get_next_event(self) -> SimEvent:
        if not self._events:
            return None
        return heapq.heappop(self._events)

    def is_empty(self) -> bool:
        return len(self._events) == 0

# --- Funções de Agendamento de Eventos ---

def schedule_next_new_batch(current_sim_time: float, fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Agenda a chegada do próximo lote de novas chamadas.
    """
    # Gerar tempo para a chegada do próximo lote
    lambda_lotes = sim_params.get("lambda_lotes_novas_chamadas")
    if lambda_lotes is None:
        print(f"[{current_sim_time:.2f}s] Erro: 'lambda_lotes_novas_chamadas' não encontrado em sim_params.")
        return # Não pode agendar sem a taxa

    time_to_next_batch = generate_batch_arrival_time(lambda_lotes)
    batch_arrival_time = current_sim_time + time_to_next_batch

    # Gerar tamanho do lote
    params_novas_chamadas = sim_params.get("B_cl_params_novas_chamadas")
    if params_novas_chamadas is None:
        print(f"[{current_sim_time:.2f}s] Erro: 'B_cl_params_novas_chamadas' não encontrado em sim_params.")
        return

    batch_size = generate_batch_size(params_novas_chamadas)

    # Incrementar contador de lotes gerados
    sim_state["total_new_batches_generated"] = sim_state.get("total_new_batches_generated", 0) + 1

    batch_id = f"LOTE_NC_{sim_state['total_new_batches_generated']}"

    event_data = {
        "id_lote": batch_id,
        "tamanho_lote": batch_size,
        "tipo_lote": "novas_chamadas"
    }

    new_batch_event = SimEvent(
        event_time=batch_arrival_time,
        event_type="CHEGADA_LOTE_NOVAS_CHAMADAS",
        event_data=event_data
    )
    fel.add_event(new_batch_event)
    print(f"[{current_sim_time:.2f}s] Agendado: CHEGADA_LOTE_NOVAS_CHAMADAS ({batch_id}, Tamanho: {batch_size}) em {batch_arrival_time:.2f}s")


def schedule_next_handover_batch(current_sim_time: float, fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Agenda a chegada do próximo lote de chamadas de handover.
    """
    lambda_lotes_ho = sim_params.get("lambda_lotes_handover")
    if lambda_lotes_ho is None:
        print(f"[{current_sim_time:.2f}s] Erro: 'lambda_lotes_handover' não encontrado em sim_params.")
        # Decide se deve parar ou continuar sem handover; por enquanto, apenas retorna.
        return

    time_to_next_batch = generate_batch_arrival_time(lambda_lotes_ho)
    batch_arrival_time = current_sim_time + time_to_next_batch

    params_handover = sim_params.get("B_cl_params_handover")
    if params_handover is None:
        print(f"[{current_sim_time:.2f}s] Erro: 'B_cl_params_handover' não encontrado em sim_params.")
        return

    batch_size = generate_batch_size(params_handover)

    sim_state["total_ho_batches_generated"] = sim_state.get("total_ho_batches_generated", 0) + 1
    batch_id = f"LOTE_HO_{sim_state['total_ho_batches_generated']}"

    event_data = {
        "id_lote": batch_id,
        "tamanho_lote": batch_size,
        "tipo_lote": "handover"
    }

    new_ho_batch_event = SimEvent(
        event_time=batch_arrival_time,
        event_type="CHEGADA_LOTE_HANDOVER", # Novo tipo de evento
        event_data=event_data
    )
    fel.add_event(new_ho_batch_event)
    print(f"[{current_sim_time:.2f}s] Agendado: CHEGADA_LOTE_HANDOVER ({batch_id}, Tamanho: {batch_size}) em {batch_arrival_time:.2f}s")


# --- Funções de Manipulação de Eventos ---

def handle_new_batch_arrival(current_sim_time: float, event_data: dict, fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Processa a chegada de um lote de novas chamadas.
    Gera chamadas individuais e as envia para handle_new_call_arrival.
    Agenda o próximo lote de novas chamadas.
    """
    id_lote = event_data.get("id_lote", "LoteDesconhecido")
    tamanho_lote = event_data.get("tamanho_lote", 0)

    print(f"[{current_sim_time:.2f}s] Evento: CHEGADA_LOTE_NOVAS_CHAMADAS - ID Lote: {id_lote}, Tamanho: {tamanho_lote}")

    for i in range(tamanho_lote):
        call_id = f"{id_lote}_CH{i+1}"
        sim_state["total_new_calls_in_batches_generated"] = sim_state.get("total_new_calls_in_batches_generated", 0) + 1

        # Criar event_data para a chamada individual
        # A classe de serviço pode vir de sim_params ou ser padrão
        individual_call_event_data = {
            "id_chamada": call_id,
            "classe_servico": "default" # Exemplo, pode ser parametrizado
            # Outros dados relevantes para a chamada individual podem ser adicionados aqui
        }

        # Chamar handle_new_call_arrival para processar cada chamada do lote.
        # Nota: handle_new_call_arrival NÃO deve mais agendar a próxima chamada.
        print(f"[{current_sim_time:.2f}s]   Processando chamada individual do lote: {call_id}")
        handle_new_call_arrival(current_sim_time, individual_call_event_data, fel, sim_params, sim_state)

    # Agendar o próximo lote de NOVAS CHAMADAS
    schedule_next_new_batch(current_sim_time, fel, sim_params, sim_state)


def handle_new_call_arrival(current_sim_time: float, event_data: dict, fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Processa a chegada de uma nova chamada individual (parte de um lote).
    NOTA: A lógica de agendamento da *próxima* nova chamada foi removida.
          Contadores de chamadas geradas são atualizados em nível de lote.
    """
    call_id = event_data.get("id_chamada", "ChamadaDesconhecida")
    print(f"[{current_sim_time:.2f}s] Evento: CHEGADA_NOVA_CHAMADA - ID: {call_id}")

    C_total = sim_params.get('C_total', sim_params.get('N', 10)) # Fallback para N, depois para 10
    N_GC = sim_params.get('N_GC', 0) # Padrão 0 se não definido
    threshold_gc = C_total - N_GC

    # A atualização de cumulative_channel_occupancy_time e time_of_last_occupancy_change
    # é feita no manipulador de lote (handle_new_batch_arrival) ANTES de iterar pelas chamadas individuais.
    # E também em handle_call_departure.

    if sim_state.get("channels_occupied", 0) < threshold_gc:
        # Aceita a nova chamada
        sim_state["channels_occupied"] = sim_state.get("channels_occupied", 0) + 1
        sim_state["total_calls_accepted"] = sim_state.get("total_calls_accepted", 0) + 1
        sim_state["n_ongoing_calls"] = sim_state.get("channels_occupied", 0)

        call_service_time = generate_service_time(sim_params)
        departure_time = current_sim_time + call_service_time
        departure_event_data = {"id_chamada": call_id, "tipo_chamada": "nova"}

        departure_event = SimEvent(
            event_time=departure_time,
            event_type="PARTIDA_CHAMADA",
            event_data=departure_event_data
        )
        fel.add_event(departure_event)
        print(f"[{current_sim_time:.2f}s]   Nova Chamada {call_id} ACEITA. Ocupação: {sim_state['channels_occupied']}/{C_total}. Limite GC: {threshold_gc}. Partida agendada para {departure_time:.2f}s.")
    else:
        # Bloqueia a nova chamada
        sim_state["total_calls_blocked"] = sim_state.get("total_calls_blocked", 0) + 1
        if sim_state.get("channels_occupied", 0) < C_total:
            print(f"[{current_sim_time:.2f}s]   Nova Chamada {call_id} BLOQUEADA (GC). Ocupação: {sim_state['channels_occupied']}/{C_total}. Limite GC: {threshold_gc}.")
        else: # channels_occupied >= C_total
            print(f"[{current_sim_time:.2f}s]   Nova Chamada {call_id} BLOQUEADA (Capacidade Total). Ocupação: {sim_state['channels_occupied']}/{C_total}. Limite GC: {threshold_gc}.")

    # O log de ocupação e a atualização de time_of_last_occupancy_change são feitos
    # no handler do lote DEPOIS do loop de processamento de chamadas individuais,
    # ou em handle_call_departure. Isso evita logs múltiplos para o mesmo instante de tempo
    # se várias chamadas individuais de um lote forem processadas sem avanço de tempo.


def handle_handover_batch_arrival(current_sim_time: float, event_data: dict, fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Processa a chegada de um lote de chamadas de handover.
    Gera chamadas individuais e tenta alocar canais.
    Agenda o próximo lote de chamadas de handover.
    """
    id_lote = event_data.get("id_lote", "LoteDesconhecidoHO")
    tamanho_lote = event_data.get("tamanho_lote", 0)
    C_total = sim_params.get('C_total', sim_params.get('N', 0)) # Usa C_total ou N

    print(f"[{current_sim_time:.2f}s] Evento: CHEGADA_LOTE_HANDOVER - ID Lote: {id_lote}, Tamanho: {tamanho_lote}")

    for i in range(tamanho_lote):
        call_id = f"{id_lote}_CH{i+1}"
        sim_state["total_ho_calls_in_batches_generated"] = sim_state.get("total_ho_calls_in_batches_generated", 0) + 1

        # Atualiza métricas de ocupação ANTES de tentar alocar canal
        previous_channels_occupied = sim_state.get("channels_occupied", 0)
        delta_t = current_sim_time - sim_state.get("time_of_last_occupancy_change", current_sim_time)
        sim_state["cumulative_channel_occupancy_time"] = sim_state.get("cumulative_channel_occupancy_time",0.0) + previous_channels_occupied * delta_t

        # Atualiza ocupação acumulada para canais não-GC
        non_gc_capacity = C_total - sim_params.get("N_GC", 0)
        occupied_non_gc_channels = min(previous_channels_occupied, non_gc_capacity)
        sim_state["cumulative_channel_occupancy_time_non_gc"] = \
            sim_state.get("cumulative_channel_occupancy_time_non_gc", 0.0) + occupied_non_gc_channels * delta_t

        if sim_state.get("channels_occupied", 0) < C_total:
            sim_state["channels_occupied"] = sim_state.get("channels_occupied", 0) + 1
            sim_state["total_ho_calls_accepted"] = sim_state.get("total_ho_calls_accepted", 0) + 1

            call_service_time = generate_service_time(sim_params)
            departure_time = current_sim_time + call_service_time
            departure_event_data = {"id_chamada": call_id, "tipo_chamada": "handover"} # Adiciona tipo para possível diferenciação na partida

            departure_event = SimEvent(
                event_time=departure_time,
                event_type="PARTIDA_CHAMADA", # Novo nome do evento
                event_data=departure_event_data
            )
            fel.add_event(departure_event)
            print(f"[{current_sim_time:.2f}s]   Chamada Handover {call_id} ACEITA. Ocupação: {sim_state['channels_occupied']}/{C_total}. Partida agendada para {departure_time:.2f}s.")
        else:
            sim_state["total_ho_calls_failed"] = sim_state.get("total_ho_calls_failed", 0) + 1
            print(f"[{current_sim_time:.2f}s]   Chamada Handover {call_id} FALHOU (sem canais). Ocupação: {sim_state['channels_occupied']}/{C_total}.")

        sim_state["occupancy_log"].append((current_sim_time, sim_state.get("channels_occupied",0)))
        sim_state["time_of_last_occupancy_change"] = current_sim_time

    # Agenda o próximo lote de Handover
    schedule_next_handover_batch(current_sim_time, fel, sim_params, sim_state)


def handle_call_departure(current_sim_time: float, event_data: dict, fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Processa a partida (fim do serviço) de uma chamada (nova ou handover).
    Libera um canal e atualiza estatísticas.
    """
    call_id = event_data.get("id_chamada", "ChamadaDesconhecida")

    # Atualiza métricas de ocupação ANTES de liberar o canal
    previous_channels_occupied = sim_state.get("channels_occupied", 0)
    delta_t = current_sim_time - sim_state.get("time_of_last_occupancy_change", current_sim_time)
    sim_state["cumulative_channel_occupancy_time"] = sim_state.get("cumulative_channel_occupancy_time",0.0) + previous_channels_occupied * delta_t

    # Atualiza ocupação acumulada para canais não-GC
    C_total = sim_params.get('C_total', sim_params.get('N', 0))
    non_gc_capacity = C_total - sim_params.get("N_GC", 0)
    occupied_non_gc_channels = min(previous_channels_occupied, non_gc_capacity)
    sim_state["cumulative_channel_occupancy_time_non_gc"] = \
        sim_state.get("cumulative_channel_occupancy_time_non_gc", 0.0) + occupied_non_gc_channels * delta_t

    if sim_state.get("channels_occupied", 0) > 0:
        sim_state["channels_occupied"] = sim_state.get("channels_occupied", 0) - 1
    else:
        # Isso não deveria acontecer em uma simulação normal se tudo estiver correto
        print(f"[{current_sim_time:.2f}s] Alerta: Partida de chamada {call_id} mas sem canais ocupados registrados.")

    sim_state["n_ongoing_calls"] = sim_state.get("channels_occupied",0) # Mantendo n_ongoing_calls sincronizado
    sim_state["total_calls_finished"] = sim_state.get("total_calls_finished", 0) + 1

    sim_state["occupancy_log"].append((current_sim_time, sim_state.get("channels_occupied",0)))
    sim_state["time_of_last_occupancy_change"] = current_sim_time

    print(f"[{current_sim_time:.2f}s] Evento: PARTIDA_CHAMADA - ID: {call_id}. Ocupação: {sim_state['channels_occupied']}/{sim_params.get('C_total', sim_params.get('N',0))}")


# --- Inicialização da Simulação ---

def initialize_simulation_events(fel: FutureEventList, sim_params: dict, sim_state: dict):
    """
    Inicializa a FEL com os primeiros eventos da simulação.
    """
    # Agendar a chegada do primeiro lote de novas chamadas
    schedule_next_new_batch(sim_state['t'], fel, sim_params, sim_state)

    # Agendar a chegada do primeiro lote de chamadas de handover
    schedule_next_handover_batch(sim_state['t'], fel, sim_params, sim_state)

# --- Funções Principais da Simulação ---

def run_simulation(sim_params, sim_state):
    fel = FutureEventList()
    initialize_simulation_events(fel, sim_params, sim_state)

    max_sim_time = sim_params.get('max_sim_time', 1000)

    print(f"\n--- Iniciando Simulação ---")
    print(f"Parâmetros: {sim_params}")
    print(f"Estado Inicial: {sim_state}")
    print(f"---------------------------\n")

    while sim_state['t'] < max_sim_time and not fel.is_empty():
        current_event = fel.get_next_event()
        if current_event is None: # Segurança, embora is_empty() deva cuidar disso
            break

        sim_state['t'] = current_event.event_time # Avança o tempo da simulação

        if sim_state['t'] > max_sim_time: # Não processar eventos além do tempo máximo
            print(f"[{sim_state['t']:.2f}s] Tempo máximo de simulação atingido. Evento {current_event.event_type} em {current_event.event_time:.2f}s ignorado.")
            break

        print(f"[{sim_state['t']:.2f}s] Processando Evento: {current_event.event_type} - Dados: {current_event.event_data}")

        if current_event.event_type == "CHEGADA_LOTE_NOVAS_CHAMADAS":
            handle_new_batch_arrival(sim_state['t'], current_event.event_data, fel, sim_params, sim_state)
        elif current_event.event_type == "CHEGADA_LOTE_HANDOVER": # Novo manipulador de lote de HO
            handle_handover_batch_arrival(sim_state['t'], current_event.event_data, fel, sim_params, sim_state)
        elif current_event.event_type == "CHEGADA_NOVA_CHAMADA": # Processa chamada individual de um lote de novas chamadas
            # Esta lógica de aceitação/rejeição precisa ser implementada aqui também, similar a handover, mas para novas chamadas.
            # Por enquanto, apenas loga. A lógica detalhada de canal (incluindo GC) será em passos futuros.
            handle_new_call_arrival(sim_state['t'], current_event.event_data, fel, sim_params, sim_state)
        elif current_event.event_type == "PARTIDA_CHAMADA": # Evento de partida renomeado
            handle_call_departure(sim_state['t'], current_event.event_data, fel, sim_params, sim_state)
        else:
            print(f"[{sim_state['t']:.2f}s] Alerta: Tipo de evento desconhecido ou não manipulado: {current_event.event_type}")

    # Cálculo final da ocupação acumulada até o max_sim_time
    # Este bloco garante que o tempo desde a última mudança de ocupação até o final da simulação seja contabilizado.
    effective_sim_end_time = min(sim_state['t'], sim_params.get('max_sim_time', 0))

    if sim_state.get("time_of_last_occupancy_change", 0) < effective_sim_end_time :
        final_delta_t = effective_sim_end_time - sim_state.get("time_of_last_occupancy_change", 0)
        current_channels_occupied = sim_state.get("channels_occupied",0)

        if final_delta_t > 0:
            sim_state["cumulative_channel_occupancy_time"] = \
                sim_state.get("cumulative_channel_occupancy_time",0.0) + current_channels_occupied * final_delta_t

            C_total = sim_params.get("C_total", 0)
            N_GC = sim_params.get("N_GC", 0)
            non_gc_capacity = C_total - N_GC
            occupied_non_gc_channels = min(current_channels_occupied, non_gc_capacity)
            sim_state["cumulative_channel_occupancy_time_non_gc"] = \
                sim_state.get("cumulative_channel_occupancy_time_non_gc", 0.0) + occupied_non_gc_channels * final_delta_t

            sim_state["occupancy_log"].append((effective_sim_end_time, current_channels_occupied))
            sim_state["time_of_last_occupancy_change"] = effective_sim_end_time # Atualiza para o tempo final

    sim_state["simulation_time"] = effective_sim_end_time # Armazena o tempo efetivo de simulação para cálculo de métricas

    print(f"\n--- Fim da Simulação (Tempo: {effective_sim_end_time:.2f}s) ---")
    # Retornar o estado final para análise, se necessário
    return sim_state


if __name__ == "__main__":
    sim_params_test = {
        'lambda_new_call': 1.0,
        'lambda_ho_call': 0.2,
        'mu': 0.5,
        'N': 5, # Reduzido para teste de GC
        'C_total': 5,          # Capacidade total de canais na célula
        'max_sim_time': 50,   # Tempo máximo de simulação reduzido para teste
        'lambda_lotes_novas_chamadas': 2.0, # Aumentar taxa para forçar mais chegadas
        'B_cl_params_novas_chamadas': {'dist': 'geom', 'p': 0.8}, # Lotes menores, mais frequentes
        'lambda_lotes_handover': 1.0,       # Aumentar taxa para forçar mais chegadas
        'B_cl_params_handover': {'dist': 'geom', 'p': 0.8},    # Lotes menores, mais frequentes
        'N_GC': 2 # Número de canais de guarda para teste
    }

    sim_state_test = {
        't': 0.0,                                      # Tempo atual da simulação
        'channels_occupied': 0,                        # Número de canais atualmente ocupados
        'n_ongoing_calls': 0,                          # Contador de chamadas em andamento (deve ser igual a channels_occupied)

        'total_new_batches_generated': 0,              # Total de lotes de novas chamadas gerados
        'total_new_calls_in_batches_generated': 0,     # Total de chamadas individuais de novos lotes geradas
        'total_calls_accepted': 0,                     # Total de novas chamadas (individuais) aceitas
        'total_calls_blocked': 0,                      # Total de novas chamadas (individuais) bloqueadas/descartadas

        'total_ho_batches_generated': 0,               # Total de lotes de handover gerados
        'total_ho_calls_in_batches_generated': 0,      # Total de chamadas individuais de lotes de handover geradas
        'total_ho_calls_accepted': 0,                  # Total de chamadas de handover aceitas
        'total_ho_calls_failed': 0,                    # Total de chamadas de handover falhadas/descartadas

        'total_calls_finished': 0,                     # Total de chamadas (qualquer tipo) que terminaram o serviço

        'time_of_last_occupancy_change': 0.0,          # Tempo do último evento que mudou a ocupação de canais
        'cumulative_channel_occupancy_time': 0.0,      # Soma ponderada de (canais_ocupados * tempo_nesse_estado)
        'cumulative_channel_occupancy_time_non_gc': 0.0, # Ocupação acumulada apenas dos canais não-GC
        'occupancy_log': []                            # Log de tuplas (tempo, canais_ocupados) para análise/depuração
    }

    # A lógica de renomeação para 'total_ho_batches_generated' de uma chave 'total_ho_calls_generated'
    # (que existia na subtask anterior) não é mais necessária pois 'total_ho_batches_generated'
    # é inicializada diretamente. A chave 'event_list' também foi removida pois a FEL é gerenciada pela classe.

    print("Starting simulation test...")
    # Passa uma cópia do dicionário de estado para evitar modificações no original global durante a execução,
    # embora em Python os dicionários sejam passados por referência (mutáveis).
    # Para garantir um estado realmente limpo para cada execução (se houvesse múltiplas execuções de run_simulation),
    # seria melhor fazer uma cópia profunda: import copy; initial_state = copy.deepcopy(sim_state_test)
    # Mas para uma única execução, passar sim_state_test diretamente é ok.
    final_state = run_simulation(sim_params_test, sim_state_test.copy()) # Usar .copy() para o estado inicial da simulação

    print("\n--- Simulation Statistics ---")
    print(f"Simulation time: {final_state.get('t', 0.0):.2f}s")
    print(f"Total new batches generated: {final_state.get('total_new_batches_generated', 0)}")
    print(f"Total new calls in batches generated: {final_state.get('total_new_calls_in_batches_generated', 0)}")
    print(f"Total calls accepted (new calls): {final_state.get('total_calls_accepted', 0)}")
    print(f"Total calls blocked (new calls): {final_state.get('total_calls_blocked', 0)}")

    print(f"Total handover batches generated: {final_state.get('total_ho_batches_generated', 0)}")
    print(f"Total handover calls in batches generated: {final_state.get('total_ho_calls_in_batches_generated', 0)}")
    print(f"Total handover calls accepted: {final_state.get('total_ho_calls_accepted', 0)}")
    print(f"Total handover calls failed: {final_state.get('total_ho_calls_failed', 0)}")

    print(f"Total calls finished (from any type): {final_state.get('total_calls_finished', 0)}")
    print(f"Number of ongoing calls at end (n_ongoing_calls): {final_state.get('n_ongoing_calls', 0)}") # Deve ser igual a channels_occupied
    print(f"Final channels occupied: {final_state.get('channels_occupied', 0)}")

    max_time_param = sim_params_test.get('max_sim_time', 1.0) # Usar uma variável diferente de max_time do escopo anterior
    if max_time_param == 0: max_time_param = 1.0
    avg_occupancy = final_state.get('cumulative_channel_occupancy_time', 0.0) / max_time_param
    print(f"Average channel occupancy: {avg_occupancy:.2f} channels over {max_time_param:.2f}s")

    # --- Cálculo e Impressão de Métricas Adicionais ---

    # P_B (Probabilidade de Bloqueio de Novas Chamadas)
    total_new_calls_generated_in_batches = final_state.get('total_new_calls_in_batches_generated', 0)
    if total_new_calls_generated_in_batches > 0:
        P_B = final_state.get('total_calls_blocked', 0) / total_new_calls_generated_in_batches
        print(f"P_B (Probabilidade de Bloqueio de Novas Chamadas): {P_B:.4f}")
    else:
        print("P_B (Probabilidade de Bloqueio de Novas Chamadas): N/A (nenhuma nova chamada individual gerada)")

    # P_F (Probabilidade de Falha de Handover)
    total_ho_calls_generated_in_batches = final_state.get('total_ho_calls_in_batches_generated', 0)
    if total_ho_calls_generated_in_batches > 0:
        P_F = final_state.get('total_ho_calls_failed', 0) / total_ho_calls_generated_in_batches
        print(f"P_F (Probabilidade de Falha de Handover): {P_F:.4f}")
    else:
        print("P_F (Probabilidade de Falha de Handover): N/A (nenhuma chamada de handover individual gerada)")

    # U_nonGC (Utilização dos Canais Não Reservados)
    C_total_params = sim_params_test.get("C_total", 0)
    N_GC_params = sim_params_test.get("N_GC", 0)
    non_gc_channel_capacity = C_total_params - N_GC_params
    effective_simulation_duration = final_state.get("simulation_time", 0.0)

    if non_gc_channel_capacity > 0 and effective_simulation_duration > 0:
        U_nonGC = final_state.get("cumulative_channel_occupancy_time_non_gc", 0.0) / (non_gc_channel_capacity * effective_simulation_duration)
        U_nonGC = min(U_nonGC, 1.0) # Garante que não exceda 1.0
        print(f"U_nonGC (Utilização dos Canais Não Reservados): {U_nonGC:.4f} (Capacidade Não-GC: {non_gc_channel_capacity} canais)")
    elif non_gc_channel_capacity <= 0:
        print(f"U_nonGC (Utilização dos Canais Não Reservados): N/A (Capacidade Não-GC é {non_gc_channel_capacity}, N_GC >= C_total)")
    else: # effective_simulation_duration == 0
        print("U_nonGC (Utilização dos Canais Não Reservados): N/A (tempo de simulação é zero)")

    print("-----------------------------")
