from math import floor
from typing import List, Tuple # Keep only one set of these imports

def calculate_qj_bpp_cs(
    C_total: int,
    num_new_call_classes: int,
    num_ho_call_classes: int,
    alpha_k_new: List[float],
    alpha_k_ho: List[float],
    b_k_new: List[int],
    b_k_ho: List[int],
    B_cl_minus_1_new: List[List[float]],
    B_cl_minus_1_ho: List[List[float]]
) -> List[float]:
    """
    Calculates the channel occupancy distribution q(j) for a system with
    Batch Poisson Process (BPP) traffic under Circuit Switching (CS),
    based on Equation 6 from the reference paper.

    This function serves as a foundational step for more complex calculations
    like blocking probabilities in systems with multi-service traffic.

    Args:
        C_total: Total capacity of the system in channels (int).
        num_new_call_classes: Number of new call classes (K) (int).
        num_ho_call_classes: Number of handover call classes (K') (int).
        alpha_k_new: List of offered load (αk) for each new call class.
                     Length must be `num_new_call_classes`. (List[float]).
        alpha_k_ho: List of offered load (αhk) for each handover call class.
                    Length must be `num_ho_call_classes`. (List[float]).
        b_k_new: List of the number of channels (bk) required by a call
                 from each new call class. Length must be `num_new_call_classes`.
                 (List[int]).
        b_k_ho: List of the number of channels (bk) required by a call
                from each handover call class. Length must be `num_ho_call_classes`.
                (List[int]).
        B_cl_minus_1_new: List of lists for the complementary batch size
                          distribution for new calls. B_cl_minus_1_new[k] is a
                          list for new call class k, where B_cl_minus_1_new[k][l-1]
                          is P(batch size of class k >= l).
                          Outer length: `num_new_call_classes`. (List[List[float]]).
        B_cl_minus_1_ho: Similar to B_cl_minus_1_new, but for handover calls.
                         Outer length: `num_ho_call_classes`. (List[List[float]]).

    Returns:
        q_j_dist: A list representing the channel occupancy distribution q(j)
                  for j = 0 to C_total. This distribution is normalized such
                  that sum(q_j_dist) = 1.0. (List[float]).
                  Returns an empty list or None if calculation is not yet implemented.
    """
    # 1. Validate Types and Primitive Values
    if not isinstance(C_total, int):
        raise ValueError(f"C_total must be an integer. Got {type(C_total)}.")
    if C_total < 0:
        raise ValueError(f"C_total must be non-negative. Got {C_total}.")

    if not isinstance(num_new_call_classes, int):
        raise ValueError(f"num_new_call_classes must be an integer. Got {type(num_new_call_classes)}.")
    if num_new_call_classes < 0:
        raise ValueError(f"num_new_call_classes must be non-negative. Got {num_new_call_classes}.")

    if not isinstance(num_ho_call_classes, int):
        raise ValueError(f"num_ho_call_classes must be an integer. Got {type(num_ho_call_classes)}.")
    if num_ho_call_classes < 0:
        raise ValueError(f"num_ho_call_classes must be non-negative. Got {num_ho_call_classes}.")

    # 2. Consistency for New Call Lists
    if not isinstance(alpha_k_new, list):
        raise ValueError(f"alpha_k_new must be a list. Got {type(alpha_k_new)}.")
    if len(alpha_k_new) != num_new_call_classes:
        raise ValueError(f"Length of alpha_k_new ({len(alpha_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")

    if not isinstance(b_k_new, list):
        raise ValueError(f"b_k_new must be a list. Got {type(b_k_new)}.")
    if len(b_k_new) != num_new_call_classes:
        raise ValueError(f"Length of b_k_new ({len(b_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")

    if not isinstance(B_cl_minus_1_new, list):
        raise ValueError(f"B_cl_minus_1_new must be a list. Got {type(B_cl_minus_1_new)}.")
    if len(B_cl_minus_1_new) != num_new_call_classes:
        raise ValueError(f"Length of B_cl_minus_1_new ({len(B_cl_minus_1_new)}) must match num_new_call_classes ({num_new_call_classes}).")

    # 3. Consistency for Handover Call Lists
    if num_ho_call_classes > 0:
        if not isinstance(alpha_k_ho, list):
            raise ValueError(f"alpha_k_ho must be a list when num_ho_call_classes > 0. Got {type(alpha_k_ho)}.")
        if len(alpha_k_ho) != num_ho_call_classes:
            raise ValueError(f"Length of alpha_k_ho ({len(alpha_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")

        if not isinstance(b_k_ho, list):
            raise ValueError(f"b_k_ho must be a list when num_ho_call_classes > 0. Got {type(b_k_ho)}.")
        if len(b_k_ho) != num_ho_call_classes:
            raise ValueError(f"Length of b_k_ho ({len(b_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")

        if not isinstance(B_cl_minus_1_ho, list):
            raise ValueError(f"B_cl_minus_1_ho must be a list when num_ho_call_classes > 0. Got {type(B_cl_minus_1_ho)}.")
        if len(B_cl_minus_1_ho) != num_ho_call_classes:
            raise ValueError(f"Length of B_cl_minus_1_ho ({len(B_cl_minus_1_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    elif num_ho_call_classes == 0:
        # If no HO classes, corresponding lists should ideally be empty or not checked for length if they are empty.
        # For simplicity, we'll ensure they are lists. If they are not empty, it's a potential misuse, but not strictly a length mismatch.
        if not isinstance(alpha_k_ho, list):
             raise ValueError(f"alpha_k_ho must be a list. Got {type(alpha_k_ho)}.")
        if not isinstance(b_k_ho, list):
             raise ValueError(f"b_k_ho must be a list. Got {type(b_k_ho)}.")
        if not isinstance(B_cl_minus_1_ho, list):
             raise ValueError(f"B_cl_minus_1_ho must be a list. Got {type(B_cl_minus_1_ho)}.")
        # Optionally, could enforce that these lists are empty if num_ho_call_classes is 0
        # if len(alpha_k_ho) != 0: raise ValueError(...)

    # 4. Detailed Validation for New Call List Elements
    for k in range(num_new_call_classes):
        if not isinstance(alpha_k_new[k], (float, int)):
            raise ValueError(f"alpha_k_new[{k}] must be a float or int. Got {type(alpha_k_new[k])}.")
        if alpha_k_new[k] < 0:
            raise ValueError(f"alpha_k_new[{k}] must be non-negative. Got {alpha_k_new[k]}.")
        
        if not isinstance(b_k_new[k], int):
            raise ValueError(f"b_k_new[{k}] must be an int. Got {type(b_k_new[k])}.")
        if b_k_new[k] <= 0:
            raise ValueError(f"b_k_new[{k}] must be positive. Got {b_k_new[k]}.")

        if not isinstance(B_cl_minus_1_new[k], list):
            raise ValueError(f"B_cl_minus_1_new[{k}] must be a list. Got {type(B_cl_minus_1_new[k])}.")
        for l_idx, val in enumerate(B_cl_minus_1_new[k]):
            if not isinstance(val, (float, int)):
                raise ValueError(f"Element at B_cl_minus_1_new[{k}][{l_idx}] must be a float or int. Got {type(val)}.")
            if val < 0:
                raise ValueError(f"Element at B_cl_minus_1_new[{k}][{l_idx}] must be non-negative. Got {val}.")

    # 5. Detailed Validation for Handover Call List Elements
    for k in range(num_ho_call_classes):
        if not isinstance(alpha_k_ho[k], (float, int)):
            raise ValueError(f"alpha_k_ho[{k}] must be a float or int. Got {type(alpha_k_ho[k])}.")
        if alpha_k_ho[k] < 0:
            raise ValueError(f"alpha_k_ho[{k}] must be non-negative. Got {alpha_k_ho[k]}.")

        if not isinstance(b_k_ho[k], int):
            raise ValueError(f"b_k_ho[{k}] must be an int. Got {type(b_k_ho[k])}.")
        if b_k_ho[k] <= 0:
            raise ValueError(f"b_k_ho[{k}] must be positive. Got {b_k_ho[k]}.")

        if not isinstance(B_cl_minus_1_ho[k], list):
            raise ValueError(f"B_cl_minus_1_ho[{k}] must be a list. Got {type(B_cl_minus_1_ho[k])}.")
        for l_idx, val in enumerate(B_cl_minus_1_ho[k]):
            if not isinstance(val, (float, int)):
                raise ValueError(f"Element at B_cl_minus_1_ho[{k}][{l_idx}] must be a float or int. Got {type(val)}.")
            if val < 0:
                raise ValueError(f"Element at B_cl_minus_1_ho[{k}][{l_idx}] must be non-negative. Got {val}.")

    # Initialize q_j_dist
    # q_j_dist will store the unnormalized q(j) values during calculation,
    # and finally the normalized distribution.
    # Size is C_total + 1 to store values for j = 0, 1, ..., C_total.
    if C_total == -1: # Should be caught by validation C_total >= 0
        q_j_dist = [] # Or handle as error, though validation should prevent this
    else:
        q_j_dist = [0.0] * (C_total + 1)

    # Set q(0) = 1.0 as the initial condition for the recurrence relation.
    # This is necessary because Equation 6 is a recurrence relation where q(j)
    # depends on previous values q(i) where i < j.
    # If C_total is 0, q_j_dist will be [1.0].
    if C_total >= 0: # Ensures q_j_dist is not empty and index 0 is valid
        q_j_dist[0] = 1.0
    
    # Placeholder for the rest of the calculation logic (Equation 6, normalization)
    # For now, returning the initialized q_j_dist with q(0)=1.0
    # This q_j_dist contains unnormalized values. Normalization will be the final step.

    # Main recursion loop to calculate q(j) for j = 1 to C_total
    # Based on Equation 6: q(j) = (1/j) * sum_{k=1 to K+K'} sum_{l=1 to floor(j/b_k)} alpha_k * b_k * P(batch size_k >= l) * q(j - l*b_k)
    for j in range(1, C_total + 1):
        sum_total_weighted_q = 0.0

        # Loop for new call classes
        for k_idx in range(num_new_call_classes):
            current_alpha_new = float(alpha_k_new[k_idx]) # Ensure float for calculations
            current_b_new = b_k_new[k_idx]
            current_B_dist_new = B_cl_minus_1_new[k_idx]

            if current_alpha_new == 0.0:
                continue

            max_l_new = floor(j / current_b_new)
            if max_l_new < 1:
                continue

            inner_sum_new = 0.0
            for l_val in range(1, max_l_new + 1):
                B_list_idx = l_val - 1
                B_prob_ge_l_new = 0.0
                if B_list_idx < len(current_B_dist_new):
                    B_prob_ge_l_new = float(current_B_dist_new[B_list_idx]) # Ensure float

                q_term_idx = j - l_val * current_b_new
                # q_term_idx must be >= 0 because l_val * current_b_new <= max_l_new * current_b_new <= (j / current_b_new) * current_b_new = j
                if q_term_idx >= 0 and q_term_idx < len(q_j_dist): # Defensive check
                     inner_sum_new += q_j_dist[q_term_idx] * B_prob_ge_l_new
                # else: Error or unexpected state, q_term_idx should be valid

            sum_total_weighted_q += current_alpha_new * current_b_new * inner_sum_new

        # Loop for handover call classes
        for k_idx in range(num_ho_call_classes):
            current_alpha_ho = float(alpha_k_ho[k_idx]) # Ensure float
            current_b_ho = b_k_ho[k_idx]
            current_B_dist_ho = B_cl_minus_1_ho[k_idx]

            if current_alpha_ho == 0.0:
                continue

            max_l_ho = floor(j / current_b_ho)
            if max_l_ho < 1:
                continue

            inner_sum_ho = 0.0
            for l_val in range(1, max_l_ho + 1):
                B_list_idx = l_val - 1
                B_prob_ge_l_ho = 0.0
                if B_list_idx < len(current_B_dist_ho):
                    B_prob_ge_l_ho = float(current_B_dist_ho[B_list_idx]) # Ensure float
                
                q_term_idx = j - l_val * current_b_ho
                if q_term_idx >= 0 and q_term_idx < len(q_j_dist): # Defensive check
                    inner_sum_ho += q_j_dist[q_term_idx] * B_prob_ge_l_ho

            sum_total_weighted_q += current_alpha_ho * current_b_ho * inner_sum_ho
        
        # Calculate q_j_dist[j]
        if j > 0: # This is always true since the loop starts from 1
            q_j_dist[j] = sum_total_weighted_q / j
        # If j is 0, this part is skipped, q_j_dist[0] remains 1.0 as set initially.

    # Normalize q_j_dist so that sum(q_j_dist) = 1.0
    total_sum_q = sum(q_j_dist)

    # Since q_j_dist[0] is initialized to 1.0 and all other terms are non-negative,
    # total_sum_q should always be >= 1.0.
    # If total_sum_q is not positive, it indicates a fundamental error in the calculation logic
    # or that q_j_dist[0] was somehow changed from its initial value of 1.0.
    if total_sum_q > 1e-9: # Use a small epsilon for floating point comparison
        q_j_dist = [val / total_sum_q for val in q_j_dist]
    else:
        # This case should ideally not be reached if q_j_dist[0] = 1.0 and other terms are non-negative.
        # It implies that either C_total < 0 (caught by validation), or q_j_dist[0] was not 1.0,
        # or subsequent calculations resulted in all zeros or negative sums, which is unexpected.
        # A special case is C_total = 0, where q_j_dist = [1.0] and total_sum_q = 1.0, handled above.
        # If all alpha_k are 0, q_j_dist will be [1.0, 0.0, ...], sum is 1.0, also handled above.
        raise RuntimeError(
            f"Normalization failed: total_sum_q is not positive ({total_sum_q}). "
            f"This indicates a critical issue with the calculation of q(j) values. "
            f"q_j_dist (unnormalized): {q_j_dist}"
        )

    return q_j_dist

if __name__ == "__main__":
    examples = [
        {
            "name": "Exemplo 1: Caso Poisson Simples (1 Erlang, C=3, b=1)",
            "params": {
                "C_total": 3,
                "num_new_call_classes": 1,
                "num_ho_call_classes": 0,
                "alpha_k_new": [1.0], # Carga total = 1.0 * E[Lote] = 1.0 * 1 = 1 Erlang
                "alpha_k_ho": [],
                "b_k_new": [1],
                "b_k_ho": [],
                "B_cl_minus_1_new": [[1.0]], # P(L>=1)=1. Implica E[Lote]=1.
                "B_cl_minus_1_ho": []
            },
            # Esperado: Erlang B para A=1, C=3. q(0) alta, q(3) baixa.
            # P_bloqueio (Erlang B) = 0.06666...
            # q(0) = 1 / (1 + 1 + 1/2 + 1/6) = 1 / (2.6666) = 0.375
            # q(1) = 1 * q(0) = 0.375
            # q(2) = (1/2) * q(0) = 0.1875
            # q(3) = (1/6) * q(0) = 0.0625
            # Soma = 0.375+0.375+0.1875+0.0625 = 1.0
        },
        {
            "name": "Exemplo 2: Caso BPP Simples (C=5, b=2)",
            "params": {
                "C_total": 5,
                "num_new_call_classes": 1,
                "num_ho_call_classes": 0,
                "alpha_k_new": [0.5], # alpha_k = lambda_batch / mu_service_time
                "alpha_k_ho": [],
                "b_k_new": [2], # cada chamada (lote) usa 2 canais
                "b_k_ho": [],
                # P(L>=1)=0.8, P(L>=2)=0.5, P(L>=3)=0.1. E[L] = 0.8+0.5+0.1 = 1.4
                # Carga total = alpha_k * E[L] * b_k = 0.5 * 1.4 * 2 = 1.4 Erlangs efetivos em termos de ocupação de canal-tempo
                "B_cl_minus_1_new": [[0.8, 0.5, 0.1]], 
                "B_cl_minus_1_ho": []
            }
        },
        {
            "name": "Exemplo 3: Caso com Múltiplas Classes de Novas Chamadas",
            "params": {
                "C_total": 4,
                "num_new_call_classes": 2,
                "num_ho_call_classes": 0,
                "alpha_k_new": [0.3, 0.2],
                "alpha_k_ho": [],
                "b_k_new": [1, 2],
                "b_k_ho": [],
                "B_cl_minus_1_new": [[1.0], [1.0, 0.5]], # Classe 1: Poisson, b=1. Classe 2: BPP, b=2, E[Lote]=1.5
                "B_cl_minus_1_ho": []
            }
        },
        {
            "name": "Exemplo 4: Caso com Chamadas de Handover",
            "params": {
                "C_total": 3,
                "num_new_call_classes": 1,
                "num_ho_call_classes": 1,
                "alpha_k_new": [0.4],
                "alpha_k_ho": [0.1],
                "b_k_new": [1],
                "b_k_ho": [1],
                "B_cl_minus_1_new": [[1.0]],
                "B_cl_minus_1_ho": [[1.0]]
            }
        },
        {
            "name": "Exemplo 5: Validação - C_total Negativo",
            "params": {
                "C_total": -1, # Inválido
                "num_new_call_classes": 1, "num_ho_call_classes": 0,
                "alpha_k_new": [1.0], "alpha_k_ho": [],
                "b_k_new": [1], "b_k_ho": [],
                "B_cl_minus_1_new": [[1.0]], "B_cl_minus_1_ho": []
            }
        },
        {
            "name": "Exemplo 6: Validação - Comprimento Incorreto de alpha_k_new",
            "params": {
                "C_total": 3,
                "num_new_call_classes": 2, # Espera 2 elementos
                "num_ho_call_classes": 0,
                "alpha_k_new": [1.0], # Fornecido 1 elemento
                "alpha_k_ho": [],
                "b_k_new": [1,1], "b_k_ho": [],
                "B_cl_minus_1_new": [[1.0],[1.0]], "B_cl_minus_1_ho": []
            }
        },
         {
            "name": "Exemplo 7: Tráfego Zero (Todas as cargas alpha são zero)",
            "params": {
                "C_total": 3,
                "num_new_call_classes": 1,
                "num_ho_call_classes": 0,
                "alpha_k_new": [0.0], 
                "alpha_k_ho": [],
                "b_k_new": [1],
                "b_k_ho": [],
                "B_cl_minus_1_new": [[1.0]], 
                "B_cl_minus_1_ho": []
            }
            # Esperado: q_j_dist = [1.0, 0.0, 0.0, 0.0]
        },
    ]

    for i, example_spec in enumerate(examples):
        print(f"\n--- {example_spec['name']} ---")
        print("Parâmetros de Entrada:")
        for key, value in example_spec['params'].items():
            print(f"  {key}: {value}")
        
        try:
            q_j_dist_result = calculate_qj_bpp_cs(**example_spec['params'])
            print("Resultados:")
            print(f"  q_j_dist: {[f'{val:.6f}' for val in q_j_dist_result]}")
            sum_q = sum(q_j_dist_result)
            print(f"  Soma de q(j): {sum_q:.6f}")
            if abs(sum_q - 1.0) > 1e-5: # Permitir pequena margem para erros de ponto flutuante
                 print(f"  AVISO: Soma de q(j) não é 1.0! Diferença: {sum_q - 1.0}")

        except Exception as e:
            print(f"  Erro durante a execução: {e}")
        if i < len(examples) - 1:
            print("-" * 50)
