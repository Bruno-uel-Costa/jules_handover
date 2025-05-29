from math import factorial

def calculate_hong_rappaport_metrics(C_total: int, C_guard: int, lambda_new: float, lambda_ho: float, avg_holding_time: float) -> tuple[float, float]:
    """
    Calculates the Hong-Rappaport metrics for a cell/beam.

    This function computes the probability of blocking new calls (P_B) and
    the probability of handover failure (P_fh) based on the Hong-Rappaport
    model for cellular systems with guard channels.

    Args:
        C_total: Total number of channels in the cell/beam (int).
        C_guard: Number of channels reserved for handover (int).
        lambda_new: Arrival rate of new calls (float).
        lambda_ho: Arrival rate of handover calls (float).
        avg_holding_time: Average channel holding time (float).

    Returns:
        A tuple containing:
            P_B: Probability of blocking new calls (float).
            P_fh: Probability of handover failure (float).
    """
    if C_total < 0:
        raise ValueError("Total number of channels (C_total) must be non-negative.")
    if C_guard < 0: # Though C_guard > C_total would catch C_guard being negative if C_total is 0.
        raise ValueError("Number of guard channels (C_guard) must be non-negative.")
    if C_guard > C_total:
        raise ValueError("Number of guard channels cannot exceed total channels.")

    C_shared = C_total - C_guard # This will be >= 0 if C_total >= C_guard

    if avg_holding_time <= 0:
        raise ValueError("Average holding time must be positive.")
    mu_H = 1 / avg_holding_time

    rho_new = lambda_new / mu_H
    rho_ho = lambda_ho / mu_H

    sum_Pj_sobre_P0 = 0.0
    Pj_sobre_P0_values = []

    for j in range(C_total + 1):
        current_Pj_sobre_P0: float
        if j == 0:
            current_Pj_sobre_P0 = 1.0
        elif 0 < j <= C_shared: # Covers C_shared = 0 as well, as this range becomes empty.
            current_Pj_sobre_P0 = (rho_new + rho_ho)**j / factorial(j)
        elif C_shared < j <= C_total:
            # This formula correctly handles C_shared = 0:
            # ((rho_new + rho_ho)**0 * rho_ho**(j - 0)) / factorial(j) = rho_ho**j / factorial(j)
            current_Pj_sobre_P0 = ((rho_new + rho_ho)**C_shared * rho_ho**(j - C_shared)) / factorial(j)
        else:
            # This case should not be reached due to the loop range and conditions.
            # If it is, it indicates a logic error in loop or conditions.
            raise ValueError(f"Unexpected value of j: {j} in Pj_sobre_P0 calculation.")

        Pj_sobre_P0_values.append(current_Pj_sobre_P0)
        sum_Pj_sobre_P0 += current_Pj_sobre_P0

    # Calculate P0: Probability of the system being in state 0 (no channels occupied).
    # sum_Pj_sobre_P0 is the sum of (Pj/P0) for all j.
    # Since Pj_sobre_P0_values[0] is 1.0 (for j=0), sum_Pj_sobre_P0 will always be >= 1.0.
    # Thus, division by zero is not possible here.
    P0 = 1 / sum_Pj_sobre_P0

    # Calculate P_values: list of actual probabilities Pj for each state j.
    P_values = [val * P0 for val in Pj_sobre_P0_values]

    # Calculate P_B: Probability of Blocking New Calls
    P_B = 0.0
    # New calls are blocked if the number of occupied channels j is >= C_shared.
    # This means summing P_values[j] from j = C_shared up to C_total.
    # The checks for C_guard > C_total and avg_holding_time <= 0 are done earlier.
    # C_shared = C_total - C_guard.
    # If C_shared < 0, it means C_guard > C_total, which is caught by the ValueError.
    # So, C_shared will be >= 0 due to C_total >= C_guard.
    # If C_shared == C_total (all channels are shared, no guard channels, C_guard = 0),
    # new calls are blocked only if all C_total channels are busy (j=C_total).
    # The loop range(C_total, C_total + 1) correctly sums P_values[C_total].
    # If C_shared == 0 (all channels are guard channels, or C_total = 0 and C_guard = 0),
    # new calls are blocked if any channel j >= 0 is occupied.
    # The loop range(0, C_total + 1) correctly sums all P_values.

    # C_shared will always be <= C_total because C_guard is non-negative.
    for j in range(C_shared, C_total + 1):
        # P_values has C_total + 1 elements, so P_values[j] is safe for j <= C_total.
        P_B += P_values[j]

    # Calculate P_fh: Probability of Handover Failure
    # Pfh is the probability that all C_total channels are busy.
    # This corresponds to P_values[C_total].
    # P_values has C_total + 1 elements, so P_values[C_total] is the last element.
    # This access is safe because C_total is validated to be non-negative and
    # P_values is populated accordingly.
    P_fh = P_values[C_total]

    return (P_B, P_fh)

if __name__ == "__main__":
    example_sets = [
        {
            "name": "Exemplo 1: Carga Moderada com Canais de Guarda",
            "params": {
                "C_total": 10,
                "C_guard": 2,
                "lambda_new": 5.0,
                "lambda_ho": 2.0,
                "avg_holding_time": 3.0
            }
        },
        {
            "name": "Exemplo 2: Alta Carga sem Canais de Guarda",
            "params": {
                "C_total": 5,
                "C_guard": 0,
                "lambda_new": 8.0,
                "lambda_ho": 1.0,
                "avg_holding_time": 2.0
            }
        },
        {
            "name": "Exemplo 3: Caso com C_total pequeno",
            "params": {
                "C_total": 2,
                "C_guard": 1,
                "lambda_new": 1.0,
                "lambda_ho": 0.5,
                "avg_holding_time": 2.0
            }
        },
        {
            "name": "Exemplo 4: Caso Inválido (C_guard > C_total)",
            "params": {
                "C_total": 5,
                "C_guard": 6,
                "lambda_new": 5.0,
                "lambda_ho": 2.0,
                "avg_holding_time": 3.0
            }
        },
        {
            "name": "Exemplo 5: Caso Inválido (avg_holding_time <= 0)",
            "params": {
                "C_total": 10,
                "C_guard": 2,
                "lambda_new": 5.0,
                "lambda_ho": 2.0,
                "avg_holding_time": 0.0
            }
        },
        {
            "name": "Exemplo 6: Tráfego Zero",
            "params": {
                "C_total": 5,
                "C_guard": 1,
                "lambda_new": 0.0,
                "lambda_ho": 0.0,
                "avg_holding_time": 2.0
            }
        },
        {
            "name": "Exemplo 7: Caso Inválido (C_total < 0)",
            "params": {
                "C_total": -1,
                "C_guard": 0,
                "lambda_new": 5.0,
                "lambda_ho": 2.0,
                "avg_holding_time": 3.0
            }
        },
        {
            "name": "Exemplo 8: Caso Inválido (C_guard < 0)",
            "params": {
                "C_total": 5,
                "C_guard": -1,
                "lambda_new": 5.0,
                "lambda_ho": 2.0,
                "avg_holding_time": 3.0
            }
        }
    ]

    for example in example_sets:
        print("-" * 40)
        print(f"Executando {example['name']}")
        print("Parâmetros de Entrada:")
        for key, value in example['params'].items():
            print(f"  {key}: {value}")
        
        try:
            P_B, P_fh = calculate_hong_rappaport_metrics(**example['params'])
            print("Resultados:")
            print(f"  Probabilidade de Bloqueio de Novas Chamadas (P_B): {P_B:.6f}")
            print(f"  Probabilidade de Falha de Handover (P_fh): {P_fh:.6f}")
        except ValueError as e:
            print(f"  Erro: {e}")
        print("-" * 40)
