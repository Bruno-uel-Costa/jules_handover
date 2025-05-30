import heapq
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass(order=True)
class SimEvent:
    event_time: float
    # Usar um contador para desempate se tempos forem iguais, 
    # heapq pode ter problemas com objetos não comparáveis.
    # Adicionar um contador de sequência simples.
    # No entanto, para simplicidade inicial, vamos omitir o contador de desempate
    # e confiar que os tipos de evento ou dados não causarão problemas de comparação.
    # Se o heapq reclamar sobre tipos não comparáveis quando event_times são iguais,
    # precisaremos adicionar um contador de sequência como segundo item na tupla de ordenação.
    event_type: str # Ordenação por event_type pode ocorrer se event_time for igual
    event_data: Dict[str, Any] = field(default_factory=dict)

class FutureEventList:
    def __init__(self):
        self.events_heap: List[SimEvent] = []

    def add_event(self, event: SimEvent) -> None:
        heapq.heappush(self.events_heap, event)

    def get_next_event(self) -> Optional[SimEvent]:
        if not self.events_heap:
            return None
        return heapq.heappop(self.events_heap)

    def is_empty(self) -> bool:
        return not self.events_heap

    def peek_next_event_time(self) -> Optional[float]:
        if not self.events_heap:
            return None
        return self.events_heap[0].event_time


# --- Funções Placeholder para Manipuladores de Eventos ---
def handle_new_call_arrival(current_time: float, event_data: Dict[str, Any], fel: FutureEventList):
    print(f"  Manipulador: handle_new_call_arrival chamado em {current_time:.4f} com dados {event_data}")
    # Exemplo de como um novo evento poderia ser agendado (não para este passo, apenas ilustrativo):
    # if "call_duration" in event_data:
    #     departure_time = current_time + event_data["call_duration"]
    #     departure_event = SimEvent(departure_time, "PARTIDA_CHAMADA", {"call_id": event_data.get("id")})
    #     fel.add_event(departure_event)
    #     print(f"    Agendado PARTIDA_CHAMADA para call_id {event_data.get('id')} em {departure_time:.4f}")

def handle_handover_arrival(current_time: float, event_data: Dict[str, Any], fel: FutureEventList):
    print(f"  Manipulador: handle_handover_arrival chamado em {current_time:.4f} com dados {event_data}")

def handle_call_departure(current_time: float, event_data: Dict[str, Any], fel: FutureEventList):
    print(f"  Manipulador: handle_call_departure chamado em {current_time:.4f} com dados {event_data}")


# --- Main Simulation Loop ---
def run_simulation(initial_fel_events: List[SimEvent], max_simulation_time: float) -> None:
    """
    Executes the main discrete-event simulation loop.

    Args:
        initial_fel_events: A list of SimEvent objects to initialize the
                            Future Event List (FEL).
        max_simulation_time: The maximum simulation time. The simulation
                             will stop if the next event's time exceeds this value.
    """
    print(f"--- Iniciando Simulação ---")
    print(f"Tempo máximo de simulação: {max_simulation_time:.2f}")

    # Inicializar o tempo da simulação e a Lista de Eventos Futuros (FEL)
    current_simulation_time: float = 0.0
    fel = FutureEventList()

    # Adicionar eventos iniciais à FEL
    if not initial_fel_events:
        print("Nenhum evento inicial fornecido para a FEL.")
    else:
        for event in initial_fel_events:
            if event.event_time < current_simulation_time:
                print(f"AVISO: Evento inicial {event.event_type} com tempo {event.event_time:.2f} "
                      f"é anterior ao tempo inicial da simulação {current_simulation_time:.2f} e será processado imediatamente.")
                # Potencialmente ajustar current_simulation_time ou lidar de outra forma
            fel.add_event(event)
        print(f"{len(initial_fel_events)} eventos iniciais adicionados à FEL.")

    # Loop principal da simulação
    event_count = 0
    while not fel.is_empty():
        next_event_time = fel.peek_next_event_time()
        if next_event_time is None: # Segurança, embora is_empty() deva cobrir
            break 
        
        if next_event_time > max_simulation_time:
            print(f"\nTempo do próximo evento ({next_event_time:.2f}) excede o tempo máximo de simulação ({max_simulation_time:.2f}).")
            break

        # Obter o próximo evento da FEL
        current_event = fel.get_next_event()
        if current_event is None: # Segurança
            break
            
        event_count += 1

        # Avançar o tempo da simulação para o tempo do evento atual
        # É crucial que o tempo só avance. Se um evento for agendado para o passado,
        # isso indica um erro na lógica de agendamento do evento.
        if current_event.event_time < current_simulation_time:
            print(f"AVISO: Evento {current_event.event_type} (ID: {event_count}) com tempo {current_event.event_time:.2f} "
                  f"ocorreu antes do tempo atual da simulação {current_simulation_time:.2f}. "
                  f"Processando, mas isso pode indicar um problema.")
            # Não se deve reverter o tempo; processar no tempo atual ou no tempo do evento.
            # Para esta simulação, vamos processar no tempo do evento, mas isso é uma anomalia.
        
        current_simulation_time = current_event.event_time

        # Renomeado event_count para processed_event_count para clareza
        print(f"\nProcessando evento #{event_count}")
        print(f"Tempo: {current_simulation_time:.4f} | Evento: {current_event.event_type} | Dados: {current_event.event_data}")

        # Chamar o handler de evento apropriado
        if current_event.event_type == "CHEGADA_NOVA_CHAMADA":
            handle_new_call_arrival(current_simulation_time, current_event.event_data, fel)
        elif current_event.event_type == "CHEGADA_HANDOVER":
            handle_handover_arrival(current_simulation_time, current_event.event_data, fel)
        elif current_event.event_type == "PARTIDA_CHAMADA":
            handle_call_departure(current_simulation_time, current_event.event_data, fel)
        else:
            print(f"  AVISO: Tipo de evento desconhecido: {current_event.event_type}")

    # Fim da simulação
    print(f"\n--- Simulação Concluída ---")
    print(f"Tempo final da simulação: {current_simulation_time:.2f}")
    print(f"Total de eventos processados: {event_count}")
    if fel.is_empty():
        print("FEL está vazia.")
    else:
        print(f"FEL não está vazia. Próximo evento agendado para: {fel.peek_next_event_time():.2f}")

    # Aqui, normalmente se retornaria ou salvaria estatísticas da simulação
    # return collected_stats

if __name__ == "__main__":
    print("--- Testando o Núcleo da Simulação ---")

    # 1. Crie uma Lista de Eventos de Exemplo
    event_nc1 = SimEvent(event_time=10.0, event_type="CHEGADA_NOVA_CHAMADA", event_data={"classe_servico": "premium", "id_chamada": "NC001"})
    event_ho1 = SimEvent(event_time=5.2, event_type="CHEGADA_HANDOVER", event_data={"id_chamada_antiga": "HO_Prev002", "id_chamada_nova": "HO002"})
    event_dep1 = SimEvent(event_time=12.5, event_type="PARTIDA_CHAMADA", event_data={"id_chamada_finalizada": "NC001_simulada"})
    event_nc2_tarde = SimEvent(event_time=101.0, event_type="CHEGADA_NOVA_CHAMADA", event_data={"id_chamada": "NC003_tarde"}) # Para testar max_simulation_time
    event_outro = SimEvent(event_time=15.0, event_type="EVENTO_ESPECIAL", event_data={"detalhe": "teste"})


    initial_events = [event_nc1, event_ho1, event_dep1, event_nc2_tarde, event_outro]
    # Adicionando um evento com o mesmo tempo para verificar a ordem de desempate (pelo tipo de evento string)
    event_nc_mesmo_tempo = SimEvent(event_time=10.0, event_type="CHEGADA_OUTRA_NOVA_CHAMADA", event_data={"id_chamada": "NC002_mesmo_tempo"})
    initial_events.append(event_nc_mesmo_tempo)


    # 2. Defina max_simulation_time
    max_sim_time = 100.0

    # 3. Chame a Função de Simulação
    run_simulation(initial_events, max_sim_time)

    print("\n--- Teste com FEL Inicial Vazia ---")
    run_simulation([], max_sim_time)

    print("\n--- Teste com Tempo Máximo de Simulação Menor que Todos os Eventos ---")
    run_simulation(initial_events, 1.0)
