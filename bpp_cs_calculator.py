import numpy as np # Added import
from math import floor
from typing import List # Keep only one set of these imports

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


def calculate_blocking_probabilities_cs(
    q_j_dist: List[float],
    C_total: int,
    b_k_new: List[int],
    b_k_ho: List[int]
) -> dict:
    """
    Calculates the blocking (and failure) probabilities for different call classes
    in a Circuit Switched (CS) system, given the channel occupancy distribution q(j).

    For CS, the failure probability P_fk is typically equal to the blocking
    probability C_bk, as calls are either accepted or rejected upon arrival
    without queuing or retries considered in this basic model.

    Args:
        q_j_dist: A list of floats representing the normalized channel
                  occupancy distribution q(j) for j = 0 to C_total.
                  (Output from calculate_qj_bpp_cs).
        C_total: Total capacity of the system in channels (int).
        b_k_new: A list of integers, where b_k_new[k] is the number of
                 channels required by a call of new call class k.
        b_k_ho: A list of integers, where b_k_ho[k] is the number of
                channels required by a call of handover call class k.

    Returns:
        A dictionary containing the blocking/failure probabilities:
        {
            'C_bk_new': List[float] - Blocking probabilities for new call classes.
                                     C_bk_new[k] is P(blocking for new call class k).
            'P_fk_new': List[float] - Failure probabilities for new call classes.
                                     Equal to C_bk_new for CS.
            'C_bk_ho': List[float]  - Blocking probabilities for handover call classes.
                                     C_bk_ho[k] is P(blocking for HO call class k).
            'P_fk_ho': List[float]  - Failure probabilities for handover call classes.
                                     Equal to C_bk_ho for CS.
        }
    """
    # Placeholder for calculation logic
    
    # 1. Validate q_j_dist
    if not isinstance(q_j_dist, list):
        raise TypeError(f"q_j_dist must be a list. Got {type(q_j_dist)}.")
    if not q_j_dist:
        raise ValueError("q_j_dist cannot be empty.")
    for i, q_val in enumerate(q_j_dist):
        if not isinstance(q_val, (float, int)):
            raise TypeError(f"All elements in q_j_dist must be float or int. Found type {type(q_val)} at index {i}.")
        if q_val < 0:
            raise ValueError(f"All elements in q_j_dist must be non-negative. Found {q_val} at index {i}.")
    
    sum_q_j = sum(q_j_dist)
    if abs(sum_q_j - 1.0) > 1e-6:
        raise ValueError(f"q_j_dist must be normalized (sum of elements must be close to 1.0). Sum is {sum_q_j}.")

    # 2. Validate C_total
    if not isinstance(C_total, int):
        raise TypeError(f"C_total must be an integer. Got {type(C_total)}.")
    if C_total < 0:
        raise ValueError(f"C_total must be non-negative. Got {C_total}.")
    if len(q_j_dist) != C_total + 1:
        raise ValueError(f"Length of q_j_dist ({len(q_j_dist)}) must be C_total + 1 ({C_total + 1}).")

    # 3. Validate b_k_new
    if not isinstance(b_k_new, list):
        raise TypeError(f"b_k_new must be a list. Got {type(b_k_new)}.")
    if C_total == 0 and b_k_new:
        # If C_total is 0, no channels are available, so no b_k > 0 can exist.
        # b_k_new should be empty in this case.
        # We check if any b_k_new[i] > 0, but since b_k must be >0, any element implies this.
        # This also covers the case where b_k_new might contain non-positive values if not for the loop below.
        # However, the main check is that if C_total is 0, b_k_new must be empty as no positive b_k can satisfy b_k <= C_total.
        if any(b > 0 for b in b_k_new): # More precise check if list isn't empty
             raise ValueError("b_k_new must be empty or contain only b_k <= 0 if C_total is 0 (as b_k must be > 0).")
        # If b_k_new is not empty but C_total is 0, the loop check `0 < b <= C_total` will fail for any positive b.
        # So, simply checking `if C_total == 0 and b_k_new:` might be too strict if b_k_new could be `[0]` (invalid by `b>0`).
        # The current loop handles individual b values correctly.
        # Let's refine this: if C_total = 0, b_k_new must be empty because b_k must be > 0.
    
    for i, b_val in enumerate(b_k_new):
        if not isinstance(b_val, int):
            raise TypeError(f"Elements in b_k_new must be integers. Found type {type(b_val)} at index {i}.")
        if C_total == 0: # If C_total is 0, no positive b_k can be valid.
            if b_val > 0:
                 raise ValueError(f"Elements in b_k_new must be <= C_total (0). Found b_k_new[{i}] = {b_val}.")
        elif not (0 < b_val <= C_total): # If C_total > 0
            raise ValueError(f"Elements in b_k_new must satisfy 0 < b <= C_total. Found b_k_new[{i}] = {b_val} with C_total = {C_total}.")


    # 4. Validate b_k_ho
    if not isinstance(b_k_ho, list):
        raise TypeError(f"b_k_ho must be a list. Got {type(b_k_ho)}.")
    if C_total == 0 and b_k_ho:
        if any(b > 0 for b in b_k_ho):
            raise ValueError("b_k_ho must be empty or contain only b_k <= 0 if C_total is 0 (as b_k must be > 0).")

    for i, b_val in enumerate(b_k_ho):
        if not isinstance(b_val, int):
            raise TypeError(f"Elements in b_k_ho must be integers. Found type {type(b_val)} at index {i}.")
        if C_total == 0:
            if b_val > 0:
                 raise ValueError(f"Elements in b_k_ho must be <= C_total (0). Found b_k_ho[{i}] = {b_val}.")
        elif not (0 < b_val <= C_total):
            raise ValueError(f"Elements in b_k_ho must satisfy 0 < b <= C_total. Found b_k_ho[{i}] = {b_val} with C_total = {C_total}.")
            
    C_bk_new_list = []
    # Calculate blocking probabilities for new call classes
    # A call of class k (requiring b_k channels) is blocked if the number of
    # occupied channels j is such that C_total - j < b_k,
    # which means j > C_total - b_k.
    # So, P_block = sum_{j = C_total - b_k + 1}^{C_total} q(j).
    for b_val in b_k_new:
        # Validations ensure 0 < b_val <= C_total (unless C_total = 0, in which case b_k_new is empty)
        current_C_bk = 0.0
        if C_total == 0: # And b_k_new is empty, loop won't run. If b_k_new was not empty, validation would fail.
             # If C_total is 0, any call requiring b_val > 0 channels is blocked with probability 1.
             # However, b_k_new must be empty if C_total = 0 due to validation b_val <= C_total.
             # This path should ideally not be complex. If b_k_new is empty, this loop is skipped.
             # If C_total = 0 and somehow b_k_new = [0] (invalid input), this logic might need thought,
             # but b_val > 0 is enforced.
             # If C_total=0, then q_j_dist = [1.0].
             # If a call needs b_val=0 (not allowed), blocking is 0.
             # If a call needs b_val=1 (not allowed if C_total=0), blocking is q_j_dist[0]=1.0.
             # The loop range below handles C_total=0 correctly if b_k_new was non-empty & valid for C_total=0 (impossible).
             # Given b_k_new is empty if C_total=0, this specific block is not strictly needed.
            pass

        # The lower bound for j is C_total - b_val + 1.
        # The upper bound for j is C_total.
        # The loop range is [lower_bound_j, C_total].
        # Example: C_total = 3, b_val = 1. Blocked if j=3. Sum q(3). lower_bound_j = 3-1+1 = 3. range(3, 4). Correct.
        # Example: C_total = 3, b_val = 2. Blocked if j=2,3. Sum q(2)+q(3). lower_bound_j = 3-2+1 = 2. range(2, 4). Correct.
        # Example: C_total = 3, b_val = 3. Blocked if j=1,2,3. Sum q(1)+q(2)+q(3). lower_bound_j = 3-3+1 = 1. range(1, 4). Correct.
        
        # Ensure lower_bound_j is not less than 0, though validation b_val <= C_total should prevent C_total - b_val + 1 < 1.
        lower_bound_j = C_total - b_val + 1
        
        # If C_total = 0, b_k_new is empty, so this loop won't run.
        # If b_k_new was not empty (e.g. b_val=0, which is invalid), then:
        # C_total=0, b_val=0 (invalid) -> lower_bound_j = 1. range(1,1) is empty. current_C_bk = 0.0. Correct.
        
        # The condition `0 < b_val <= C_total` implies `lower_bound_j >= 1`.
        # Max value for `lower_bound_j` is `C_total` (when `b_val = 1`).
        # Min value for `lower_bound_j` is `1` (when `b_val = C_total`).
        
        for j_idx in range(lower_bound_j, C_total + 1):
            if 0 <= j_idx < len(q_j_dist): # Defensive check, should always be true
                current_C_bk += q_j_dist[j_idx]
            # else: this would indicate an issue with lower_bound_j or C_total relative to q_j_dist
        C_bk_new_list.append(current_C_bk)

    P_fk_new_list = list(C_bk_new_list) # For CS, P_fk = C_bk

    C_bk_ho_list = []
    # Calculate blocking probabilities for handover call classes
    # Logic is identical to new calls, just using b_k_ho
    for b_val in b_k_ho:
        current_C_bk = 0.0
        # Validations ensure 0 < b_val <= C_total (unless C_total = 0, then b_k_ho is empty)
        # If C_total == 0, this loop is skipped as b_k_ho would be empty.
        
        lower_bound_j = C_total - b_val + 1
        
        for j_idx in range(lower_bound_j, C_total + 1):
            if 0 <= j_idx < len(q_j_dist): # Defensive check
                current_C_bk += q_j_dist[j_idx]
        C_bk_ho_list.append(current_C_bk)

    P_fk_ho_list = list(C_bk_ho_list) # For CS, P_fk = C_bk

    return {
        'C_bk_new': C_bk_new_list, 
        'P_fk_new': P_fk_new_list, 
        'C_bk_ho': C_bk_ho_list, 
        'P_fk_ho': P_fk_ho_list
    }

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
        print("Parâmetros de Entrada (para calculate_qj_bpp_cs):")
        for key, value in example_spec['params'].items():
            print(f"  {key}: {value}")
        
        q_j_dist_result = None
        try:
            q_j_dist_result = calculate_qj_bpp_cs(**example_spec['params'])
            print("Resultados (calculate_qj_bpp_cs):")
            print(f"  q_j_dist: {[f'{val:.6f}' for val in q_j_dist_result]}")
            sum_q = sum(q_j_dist_result)
            print(f"  Soma de q(j): {sum_q:.6f}")
            if abs(sum_q - 1.0) > 1e-5: # Permitir pequena margem para erros de ponto flutuante
                 print(f"  AVISO: Soma de q(j) não é 1.0! Diferença: {sum_q - 1.0}")

        except Exception as e:
            print(f"  Erro durante a execução de calculate_qj_bpp_cs: {e}")
        
        # Testar calculate_blocking_probabilities_cs se q_j_dist foi calculado com sucesso
        # Usaremos o Exemplo 1 e Exemplo 4 para testar calculate_blocking_probabilities_cs
        if q_j_dist_result and example_spec["name"] in [
            "Exemplo 1: Caso Poisson Simples (1 Erlang, C=3, b=1)",
            "Exemplo 4: Caso com Chamadas de Handover"
            ]:
            print(f"\n  --- Testando calculate_blocking_probabilities_cs com saída de '{example_spec['name']}' ---")
            pb_params = {
                "q_j_dist": q_j_dist_result,
                "C_total": example_spec['params']['C_total'],
                "b_k_new": example_spec['params']['b_k_new'],
                "b_k_ho": example_spec['params']['b_k_ho']
            }
            print("  Parâmetros de Entrada (para calculate_blocking_probabilities_cs):")
            print(f"    C_total: {pb_params['C_total']}")
            print(f"    b_k_new: {pb_params['b_k_new']}")
            print(f"    b_k_ho: {pb_params['b_k_ho']}")
            # q_j_dist é muito longa para imprimir aqui novamente

            try:
                blocking_probs = calculate_blocking_probabilities_cs(**pb_params)
                print("  Resultados (calculate_blocking_probabilities_cs):")
                for key, value in blocking_probs.items():
                    if isinstance(value, list):
                        print(f"    {key}: {[f'{v:.6f}' for v in value]}")
                    else:
                        print(f"    {key}: {value}")
            except Exception as e_pb:
                print(f"    Erro durante a execução de calculate_blocking_probabilities_cs: {e_pb}")

        if i < len(examples) - 1:
            print("-" * 50)

    # Testes Adicionais para calculate_blocking_probabilities_cs
    print("\n\n--- Testes Adicionais para calculate_blocking_probabilities_cs ---")

    # 1. Exemplo Manual Simples
    print("\n--- Teste Manual Simples para calculate_blocking_probabilities_cs ---")
    C_total_manual = 3
    q_j_dist_manual = [0.1, 0.2, 0.3, 0.4] # Soma = 1.0
    b_k_new_manual = [1, 2, 3]
    b_k_ho_manual = [1]
    print(f"  Parâmetros: C_total={C_total_manual}, q_j_dist (soma)={sum(q_j_dist_manual):.1f}, b_k_new={b_k_new_manual}, b_k_ho={b_k_ho_manual}")
    # C_bk_new[0] (b=1): q[3] = 0.4
    # C_bk_new[1] (b=2): q[2]+q[3] = 0.3+0.4 = 0.7
    # C_bk_new[2] (b=3): q[1]+q[2]+q[3] = 0.2+0.3+0.4 = 0.9
    # C_bk_ho[0] (b=1): q[3] = 0.4
    try:
        blocking_probs_manual = calculate_blocking_probabilities_cs(q_j_dist_manual, C_total_manual, b_k_new_manual, b_k_ho_manual)
        print("  Resultados:")
        for key, value in blocking_probs_manual.items():
            if isinstance(value, list):
                print(f"    {key}: {[f'{v:.6f}' for v in value]}")
            else:
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"  Erro: {e}")
    print("-" * 50)


    # --- Testes para calculate_qj_bpp_br ---
    print("\n\n--- Testes para calculate_qj_bpp_br ---")

    # 1. Teste de Comparação com calculate_qj_bpp_cs (todos t_k = 0)
    # Com a política j <= C_total - t_k, t_k=0 significa j <= C_total, que é CS.
    print("\n--- Teste de Comparação BR vs CS (t_k = 0) ---")
    # Usar Exemplo 1 (Poisson Simples) de calculate_qj_bpp_cs
    cs_example_params_for_comp = examples[0]["params"] # Renomeado para evitar conflito
    C_total_comp = cs_example_params_for_comp["C_total"]
    num_new_comp = cs_example_params_for_comp["num_new_call_classes"]
    num_ho_comp = cs_example_params_for_comp["num_ho_call_classes"]

    # Para BR se comportar como CS (j <= C_total - t_k), t_k deve ser 0.
    t_k_new_zeros = [0] * num_new_comp
    t_k_ho_zeros = [0] * num_ho_comp
    
    br_params_comp = {**cs_example_params_for_comp, "t_k_new": t_k_new_zeros, "t_k_ho": t_k_ho_zeros}

    print("Parâmetros para BR (t_k = 0):")
    for key, value in br_params_comp.items():
            print(f"  {key}: {value}")
    try:
        q_br_comp = calculate_qj_bpp_br(**br_params_comp)
        print(f"  q_j_dist (BR, t_k=0): {[f'{val:.6f}' for val in q_br_comp]}")
        
        q_cs_comp = calculate_qj_bpp_cs(**cs_example_params_for_comp)
        print(f"  q_j_dist (CS):      {[f'{val:.6f}' for val in q_cs_comp]}")

        diff_sum_qj = sum(abs(q_br_comp[i] - q_cs_comp[i]) for i in range(len(q_br_comp)))
        print(f"  Soma das diferenças absolutas q(j): {diff_sum_qj:.9f}")
        
        qj_match = diff_sum_qj < 1e-9
        if qj_match:
            print("  SUCESSO: Distribuições BR (t_k=0) e CS são idênticas.")
            
            print("\n  --- Comparando Probabilidades de Bloqueio (BR com t_k=0 vs CS) ---")
            pb_br_params = {
                "C_total": br_params_comp["C_total"],
                "q_j_dist": q_br_comp,
                "b_k_new": br_params_comp["b_k_new"],
                "b_k_ho": br_params_comp["b_k_ho"],
                "t_k_new": br_params_comp["t_k_new"], # Lista de zeros
                "t_k_ho": br_params_comp["t_k_ho"]  # Lista de zeros
            }
            pb_br_results = calculate_blocking_probabilities_br(**pb_br_params)
            
            pb_cs_params = {
                "C_total": cs_example_params_for_comp["C_total"],
                "q_j_dist": q_cs_comp,
                "b_k_new": cs_example_params_for_comp["b_k_new"],
                "b_k_ho": cs_example_params_for_comp["b_k_ho"]
            }
            pb_cs_results_dict = calculate_blocking_probabilities_cs(**pb_cs_params)
            # Adaptar chaves do pb_cs_results_dict para corresponder às de pb_br_results para comparação
            pb_cs_results_adapted = {
                'P_B_new': pb_cs_results_dict.get('C_bk_new', []), # Em CS, C_bk = P_B
                'P_B_handover': pb_cs_results_dict.get('C_bk_ho', [])
            }

            print(f"    Resultados de Probabilidade de Bloqueio (BR com t_k=0):")
            for key, value in pb_br_results.items():
                print(f"      {key}: {[f'{v:.6f}' for v in value]}")
            
            print(f"    Resultados de Probabilidade de Bloqueio (CS):")
            for key, value in pb_cs_results_adapted.items():
                 print(f"      {key}: {[f'{v:.6f}' for v in value]}")

            # Comparar os dicionários
            pb_match = True
            if len(pb_br_results['P_B_new']) != len(pb_cs_results_adapted['P_B_new']) or \
               len(pb_br_results['P_B_handover']) != len(pb_cs_results_adapted['P_B_handover']):
                pb_match = False
            else:
                for i in range(len(pb_br_results['P_B_new'])):
                    if abs(pb_br_results['P_B_new'][i] - pb_cs_results_adapted['P_B_new'][i]) > 1e-9:
                        pb_match = False
                        break
                if pb_match:
                    for i in range(len(pb_br_results['P_B_handover'])):
                        if abs(pb_br_results['P_B_handover'][i] - pb_cs_results_adapted['P_B_handover'][i]) > 1e-9:
                            pb_match = False
                            break
            
            if pb_match:
                print("    SUCESSO: Probabilidades de Bloqueio BR (t_k=0) e CS são idênticas.")
            else:
                print("    FALHA: Probabilidades de Bloqueio BR (t_k=0) e CS diferem.")
        else:
            print("  FALHA: Distribuições q(j) BR (t_k=0) e CS diferem, P_B não será comparado.")
            
    except Exception as e:
        print(f"  Erro durante o teste de comparação: {e}")
    print("-" * 50)

    # 2. Teste com Reservas Ativas (t_k > 0)
    print("\n--- Teste BR com Reservas Ativas (t_k > 0) ---")
    C_total_br_active = 5
    # t_k_new_br_active_val é o número de canais reservados CONTRA esta classe.
    # Se t_k=2, a classe pode usar canais até j <= C_total - 2 = 5 - 2 = 3.
    t_k_new_br_active_val = 2 
    
    params_br_active = {
        "C_total": C_total_br_active,
        "num_new_call_classes": 1,
        "num_ho_call_classes": 0,
        "alpha_k_new": [1.0],
        "b_k_new": [1],
        "B_cl_minus_1_new": [[1.0]], # Poisson
        "alpha_k_ho": [],
        "b_k_ho": [],
        "B_cl_minus_1_ho": [],
        "t_k_new": [t_k_new_br_active_val], 
        "t_k_ho": []
    }
    print("Parâmetros para BR (ativa):")
    for key, value in params_br_active.items():
            print(f"  {key}: {value}")
    try:
        q_br_active = calculate_qj_bpp_br(**params_br_active)
        print(f"  Resultados (calculate_qj_bpp_br, t_k={t_k_new_br_active_val}):")
        print(f"    q_j_dist: {[f'{val:.6f}' for val in q_br_active]}")
        print(f"    Soma de q(j): {sum(q_br_active):.6f}")

        admissible_threshold = C_total_br_active - t_k_new_br_active_val
        print(f"    (Para esta classe, estados j > {admissible_threshold} não são permitidos pela reserva t_k={t_k_new_br_active_val})")

        if len(q_br_active) > admissible_threshold + 1:
            are_higher_states_zero = all(abs(q_br_active[j_idx]) < 1e-9 for j_idx in range(admissible_threshold + 1, len(q_br_active)))
            if are_higher_states_zero:
                print(f"    VERIFICADO: q(j) é zero para j > {admissible_threshold} como esperado para este caso de classe única.")
            else:
                print(f"    NOTA: q(j) não é zero para j > {admissible_threshold}. Isso é esperado se houver outras classes com diferentes t_k.")

        print(f"\n  --- Testando calculate_blocking_probabilities_br com saída de 'Teste BR com Reservas Ativas' ---")
        pb_br_active_params = {
            "C_total": params_br_active["C_total"],
            "q_j_dist": q_br_active,
            "b_k_new": params_br_active["b_k_new"],
            "b_k_ho": params_br_active["b_k_ho"],
            "t_k_new": params_br_active["t_k_new"],
            "t_k_ho": params_br_active["t_k_ho"]
        }
        pb_br_active_results = calculate_blocking_probabilities_br(**pb_br_active_params)
        print(f"  Resultados de Probabilidade de Bloqueio (BR com t_k > 0):")
        for key, value in pb_br_active_results.items():
            print(f"    {key}: {[f'{v:.6f}' for v in value]}")

    except Exception as e:
        print(f"  Erro durante o teste BR com reservas ativas: {e}")
    print("-" * 50)

    # 3. Testes de Validação para t_k_new e t_k_ho
    base_params_val_br = {
        "C_total": 3,
        "num_new_call_classes": 1,
        "num_ho_call_classes": 0,
        "alpha_k_new": [1.0], "b_k_new": [1], "B_cl_minus_1_new": [[1.0]],
        "alpha_k_ho": [], "b_k_ho": [], "B_cl_minus_1_ho": []
    }

    print("\n--- Teste de Validação BR: Comprimento Incorreto de t_k_new ---")
    # C_total_br_active (5) é usado aqui para t_k_new, mas base_params_val_br.C_total (3) é usado para o cálculo.
    # Isso pode ser confuso. Vamos usar C_total de base_params_val_br para consistência no teste de validação.
    params_val_len = {**base_params_val_br, "t_k_new": [base_params_val_br['C_total'], base_params_val_br['C_total']], "t_k_ho": []} # num_new_call_classes é 1, t_k_new tem len 2
    print(f"  Parâmetros: C_total={base_params_val_br['C_total']}, num_new_call_classes={params_val_len['num_new_call_classes']}, t_k_new={params_val_len['t_k_new']}")
    try:
        calculate_qj_bpp_br(**params_val_len)
        print("  FALHA no teste de validação: ValueError (comprimento t_k_new) não foi levantado.")
    except ValueError as e:
        print(f"  SUCESSO: Teste de validação (comprimento t_k_new) falhou como esperado: {e}")
    except Exception as e:
        print(f"  FALHA no teste de validação: Exceção inesperada: {e}")
    print("-" * 50)

    print("\n--- Teste de Validação BR: Valor Negativo em t_k_new ---")
    params_val_neg = {**base_params_val_br, "t_k_new": [-1], "t_k_ho": []}
    print(f"  Parâmetros: t_k_new={params_val_neg['t_k_new']}")
    try:
        calculate_qj_bpp_br(**params_val_neg)
        print("  FALHA no teste de validação: ValueError (t_k_new negativo) não foi levantado.")
    except ValueError as e:
        print(f"  SUCESSO: Teste de validação (t_k_new negativo) falhou como esperado: {e}")
    except Exception as e:
        print(f"  FALHA no teste de validação: Exceção inesperada: {e}")
    print("-" * 50)

    print("\n--- Teste de Validação BR: Valor t_k_ho > C_total ---")
    params_val_ho_gt = {
        **base_params_val_br, 
        "num_new_call_classes":0, "alpha_k_new": [], "b_k_new": [], "B_cl_minus_1_new": [], "t_k_new": [],
        "num_ho_call_classes": 1, "alpha_k_ho": [1.0], "b_k_ho": [1], "B_cl_minus_1_ho": [[1.0]], 
        "t_k_ho": [base_params_val_br["C_total"] + 1] # t_k_ho > C_total
    }
    print(f"  Parâmetros: C_total={params_val_ho_gt['C_total']}, num_ho_call_classes={params_val_ho_gt['num_ho_call_classes']}, t_k_ho={params_val_ho_gt['t_k_ho']}")
    try:
        calculate_qj_bpp_br(**params_val_ho_gt)
        print("  FALHA no teste de validação: ValueError (t_k_ho > C_total) não foi levantado.")
    except ValueError as e:
        print(f"  SUCESSO: Teste de validação (t_k_ho > C_total) falhou como esperado: {e}")
    except Exception as e:
        print(f"  FALHA no teste de validação: Exceção inesperada: {e}")
    print("-" * 50)


def calculate_blocking_probabilities_br(
    C_total: int,
    q_j_dist: List[float],
    b_k_new: List[int],
    b_k_ho: List[int],
    t_k_new: List[int],
    t_k_ho: List[int]
) -> dict:
    """
    Calculates the blocking probabilities for different call classes in a system
    with a Trunk Reservation (BR) policy, given the channel occupancy 
    distribution q(j).

    The blocking condition for a class k call (new or handover) requiring b_k 
    channels, with a reservation threshold t_k (number of channels reserved for 
    higher priority calls), is that the call is blocked if the number of 
    currently occupied channels j is such that j + b_k > C_total - t_k.

    Args:
        C_total: Total capacity of the system in channels (int).
        q_j_dist: A list of floats representing the normalized channel
                  occupancy distribution q(j) for j = 0 to C_total.
                  (Typically the output from calculate_qj_bpp_br).
        b_k_new: A list of integers, where b_k_new[k] is the number of
                 channels required by a call of new call class k.
        b_k_ho: A list of integers, where b_k_ho[k] is the number of
                channels required by a call of handover call class k.
        t_k_new: List of trunk reservation parameters (tk) for new call classes.
                 tk is the number of channels reserved FOR OTHER (higher priority) 
                 classes. A call of new class k is subject to blocking based on
                 C_total - t_k_new[k] as its effective capacity.
                 Length must match b_k_new.
        t_k_ho: Similar for handover call classes. Length must match b_k_ho.

    Returns:
        A dictionary containing the blocking probabilities:
        {
            'P_B_new': List[float] - Blocking probabilities for new call classes.
                                   P_B_new[k] is P(blocking for new call class k).
            'P_B_handover': List[float] - Blocking probabilities for handover call classes.
                                        P_B_handover[k] is P(blocking for HO class k).
        }
        Note: In many models, for circuit-switched systems without retries/queuing,
        failure probability P_fk is the same as blocking probability P_Bk.
        This function returns P_B (blocking probability).
    """
    # Placeholder for calculation logic
    
    # 1. Validate q_j_dist
    if not isinstance(q_j_dist, list):
        raise TypeError(f"q_j_dist must be a list. Got {type(q_j_dist)}.")
    # C_total >= 0 is validated next. If C_total = 0, len(q_j_dist) should be 1.
    # If C_total >= 0, then q_j_dist cannot be empty if len(q_j_dist) == C_total + 1.
    
    for i, q_val in enumerate(q_j_dist):
        if not isinstance(q_val, (float, int)):
            raise TypeError(f"All elements in q_j_dist must be float or int. Found type {type(q_val)} at index {i}.")
        if q_val < 0: # Probabilities cannot be negative
            raise ValueError(f"All elements in q_j_dist must be non-negative. Found {q_val} at index {i}.")
    
    if q_j_dist: # Only sum if not empty, though other checks should ensure it's not empty if C_total >=0
        sum_q_j = sum(q_j_dist)
        if abs(sum_q_j - 1.0) > 1e-6:
            raise ValueError(f"q_j_dist must be normalized (sum of elements must be close to 1.0). Sum is {sum_q_j}.")
    elif C_total >= 0 : # If C_total >=0, q_j_dist should not be empty.
        raise ValueError("q_j_dist is empty but C_total >= 0, which implies q_j_dist should have C_total+1 elements.")


    # 2. Validate C_total
    if not isinstance(C_total, int):
        raise TypeError(f"C_total must be an integer. Got {type(C_total)}.")
    if C_total < 0:
        raise ValueError(f"C_total must be non-negative. Got {C_total}.")
    if len(q_j_dist) != C_total + 1:
        raise ValueError(f"Length of q_j_dist ({len(q_j_dist)}) must be C_total + 1 ({C_total + 1}).")
    # This also implies q_j_dist is not empty if C_total >= 0.

    # 3. Validate Types of Lists for b_k and t_k
    if not isinstance(b_k_new, list):
        raise TypeError(f"b_k_new must be a list. Got {type(b_k_new)}.")
    if not isinstance(b_k_ho, list):
        raise TypeError(f"b_k_ho must be a list. Got {type(b_k_ho)}.")
    if not isinstance(t_k_new, list):
        raise TypeError(f"t_k_new must be a list. Got {type(t_k_new)}.")
    if not isinstance(t_k_ho, list):
        raise TypeError(f"t_k_ho must be a list. Got {type(t_k_ho)}.")

    # 4. Validate Consistency of Length for b_k and t_k pairs
    if len(b_k_new) != len(t_k_new):
        raise ValueError(f"Length of b_k_new ({len(b_k_new)}) must match length of t_k_new ({len(t_k_new)}).")
    if len(b_k_ho) != len(t_k_ho):
        raise ValueError(f"Length of b_k_ho ({len(b_k_ho)}) must match length of t_k_ho ({len(t_k_ho)}).")

    # Note: Detailed validation of elements within b_k_new, b_k_ho, t_k_new, t_k_ho
    # (e.g., 0 < b_k <= C_total - t_k, and 0 <= t_k <= C_total)
    # will be done during the calculation phase or could be added here if preferred.
    # For now, focusing on the essential structural validations.
    # The problem description for this step only asked for these "essential" validations.

    # Adicionar validações detalhadas para elementos de b_k e t_k
    # Validação de b_k_new
    for i, b_val in enumerate(b_k_new):
        if not isinstance(b_val, int):
            raise TypeError(f"Elements in b_k_new must be integers. Found type {type(b_val)} at index {i}.")
        if b_val <= 0: # b_k deve ser positivo
            raise ValueError(f"Elements in b_k_new must be positive. Found {b_val} at index {i}.")
        # Não podemos validar b_val <= C_total - t_k_new[i] aqui sem t_k_new[i],
        # mas a lógica de bloqueio tratará casos onde b_val é muito grande.

    # Validação de t_k_new
    for i, t_val in enumerate(t_k_new):
        if not isinstance(t_val, int):
            raise TypeError(f"Elements in t_k_new must be integers. Found type {type(t_val)} at index {i}.")
        if not (0 <= t_val <= C_total):
             raise ValueError(f"Elements in t_k_new must satisfy 0 <= t_val <= C_total. Found t_k_new[{i}] = {t_val} with C_total = {C_total}.")
    
    # Validação de b_k_ho
    for i, b_val in enumerate(b_k_ho):
        if not isinstance(b_val, int):
            raise TypeError(f"Elements in b_k_ho must be integers. Found type {type(b_val)} at index {i}.")
        if b_val <= 0:
            raise ValueError(f"Elements in b_k_ho must be positive. Found {b_val} at index {i}.")

    # Validação de t_k_ho
    for i, t_val in enumerate(t_k_ho):
        if not isinstance(t_val, int):
            raise TypeError(f"Elements in t_k_ho must be integers. Found type {type(t_val)} at index {i}.")
        if not (0 <= t_val <= C_total):
            raise ValueError(f"Elements in t_k_ho must satisfy 0 <= t_val <= C_total. Found t_k_ho[{i}] = {t_val} with C_total = {C_total}.")


    P_B_new_list = []
    # Blocking condition for class k: j_occupied + b_k > C_total - t_k
    # This means blocking occurs if j_occupied > C_total - t_k - b_k.
    # So, sum q(j) from floor(C_total - t_k - b_k) + 1 up to C_total.
    # Let threshold = C_total - t_k - b_k. We sum for j > threshold.
    # lower_sum_idx = floor(threshold) + 1.
    # If threshold < 0 (i.e., C_total - t_k < b_k), it means call is blocked even if j=0. P_B = 1.
    # In this case, lower_sum_idx will be <= 0. range(max(0, lower_sum_idx), C_total + 1) sums all q(j).
    
    for b_val, t_k_val in zip(b_k_new, t_k_new):
        # Effective capacity for this class: C_eff = C_total - t_k_val
        # Call is blocked if j_occupied + b_val > C_eff
        # Or, j_occupied > C_eff - b_val
        # Sum q(j) for j from floor(C_eff - b_val) + 1 to C_total
        
        # Check if b_val itself is compatible with the effective capacity C_total - t_k_val
        if b_val > (C_total - t_k_val): # If channels needed > available channels after reservation
            current_P_B = 1.0 # Always blocked
        else:
            lower_bound_j_exclusive = C_total - t_k_val - b_val
            # We need to sum for j > lower_bound_j_exclusive
            # So, j starts from floor(lower_bound_j_exclusive) + 1
            # which is equivalent to (C_total - t_k_val - b_val) + 1, if integer, or floor+1
            
            # Simpler: sum q(j) where j + b_val > C_total - t_k_val
            # The first j for which this is NOT blocked is when j + b_val <= C_total - t_k_val
            # So, j_max_non_blocked = C_total - t_k_val - b_val
            # Any j > j_max_non_blocked will be blocked.
            # So, sum from j_max_non_blocked + 1 to C_total
            
            start_sum_idx = (C_total - t_k_val - b_val) + 1
            current_P_B = 0.0
            
            # The loop should sum q(j) for j from max(0, start_sum_idx) to C_total
            for j_idx in range(max(0, start_sum_idx), C_total + 1):
                # Defensive check, though q_j_dist length is C_total + 1
                if j_idx < len(q_j_dist): # Should always be true
                    current_P_B += q_j_dist[j_idx]
        P_B_new_list.append(current_P_B)

    P_B_ho_list = []
    # Logic is identical to P_B_new_list, but using b_k_ho and t_k_ho
    for b_val, t_k_val in zip(b_k_ho, t_k_ho):
        # Validations for b_val > 0 and 0 <= t_k_val <= C_total already done above.
        
        if b_val > (C_total - t_k_val): # If channels needed > available channels after reservation
            current_P_B = 1.0 # Always blocked
        else:
            start_sum_idx = (C_total - t_k_val - b_val) + 1
            current_P_B = 0.0
            
            for j_idx in range(max(0, start_sum_idx), C_total + 1):
                if j_idx < len(q_j_dist): # Should always be true
                    current_P_B += q_j_dist[j_idx]
        P_B_ho_list.append(current_P_B)

    return {'P_B_new': P_B_new_list, 'P_B_handover': P_B_ho_list}


def calculate_qj_bpp_br(
    C_total: int,
    num_new_call_classes: int,
    num_ho_call_classes: int,
    alpha_k_new: List[float],
    alpha_k_ho: List[float],
    b_k_new: List[int],
    b_k_ho: List[int],
    B_cl_minus_1_new: List[List[float]],
    B_cl_minus_1_ho: List[List[float]],
    t_k_new: List[int], # Trunk reservation thresholds for new calls
    t_k_ho: List[int]  # Trunk reservation thresholds for HO calls
) -> List[float]:
    """
    Calculates the channel occupancy distribution q(j) for a system with
    Batch Poisson Process (BPP) traffic under a Trunk Reservation (BR) policy.

    In this policy, a call of class k (new or handover) requiring b_k channels
    is accepted if the number of currently occupied channels j_prev satisfies
    j_prev + b_k <= C_total - t_k. Here, t_k is the number of channels
    reserved for higher priority calls (typically t_k is 0 for the highest
    priority, often handover calls, and increases for lower priority calls).
    The condition means a call is accepted if it uses channels up to
    C_total - t_k.

    The calculation is based on a recurrence relation:
    j * q(j) = sum_{all classes k} sum_{l=1 to floor(j/b_k)}
               alpha_k * b_k * P(batch_size_k >= l) * q(j - l*b_k) * I_k(j)
    where I_k(j) is an indicator function that is 1 if a call of class k
    leading to state j (i.e., j_prev + l*b_k = j) would be accepted, and 0 otherwise.
    The acceptance condition is j <= C_total - t_k_class[k_idx].

    Args:
        C_total: Total capacity of the system in channels (int).
        num_new_call_classes: Number of new call classes (K) (int).
        num_ho_call_classes: Number of handover call classes (K') (int).
        alpha_k_new: List of offered load (αk) for each new call class.
        alpha_k_ho: List of offered load (αhk) for each handover call class.
        b_k_new: List of channels (bk) required by each new call class.
        b_k_ho: List of channels (bk) required by each handover call class.
        B_cl_minus_1_new: Complementary batch size distribution for new calls.
                          B_cl_minus_1_new[k][l-1] is P(batch size of class k >= l).
        B_cl_minus_1_ho: Similar for handover calls.
        t_k_new: List of trunk reservation parameters (tk) for new call classes.
                 tk is the number of channels reserved FOR OTHER (higher priority) classes.
                 A call of new class k is accepted if j_occupied_after_acceptance <= C_total - t_k_new[k].
                 Constraint: 0 <= t_k_new[k] <= C_total.
                 If t_k_new[k] = 0, it means no channels are reserved against this class (it can use up to C_total).
                 If t_k_new[k] = C_total, it means the class can only be accepted if j_occupied_after_acceptance <= 0.
                 Length must be `num_new_call_classes`. (List[int]).
        t_k_ho: Similar for handover call classes.
                Constraint: 0 <= t_k_ho[k] <= C_total.
                Length must be `num_ho_call_classes`. (List[int]).

    Returns:
        q_j_dist: Normalized channel occupancy distribution q(j). (List[float]).
    """
    # --- Validações de Entrada ---
    # 1. Validate Types and Primitive Values (C_total, num_new_call_classes, num_ho_call_classes)
    if not isinstance(C_total, int):
        raise TypeError(f"C_total must be an integer. Got {type(C_total)}.")
    if C_total < 0:
        raise ValueError(f"C_total must be non-negative. Got {C_total}.")

    if not isinstance(num_new_call_classes, int):
        raise TypeError(f"num_new_call_classes must be an integer. Got {type(num_new_call_classes)}.")
    if num_new_call_classes < 0:
        raise ValueError(f"num_new_call_classes must be non-negative. Got {num_new_call_classes}.")

    if not isinstance(num_ho_call_classes, int):
        raise TypeError(f"num_ho_call_classes must be an integer. Got {type(num_ho_call_classes)}.")
    if num_ho_call_classes < 0:
        raise ValueError(f"num_ho_call_classes must be non-negative. Got {num_ho_call_classes}.")

    # 2. Consistency and Content for New Call Lists (alpha_k_new, b_k_new, B_cl_minus_1_new, t_k_new)
    if not isinstance(alpha_k_new, list):
        raise TypeError(f"alpha_k_new must be a list. Got {type(alpha_k_new)}.")
    if len(alpha_k_new) != num_new_call_classes:
        raise ValueError(f"Length of alpha_k_new ({len(alpha_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, val in enumerate(alpha_k_new):
        if not isinstance(val, (float, int)):
            raise TypeError(f"Elements in alpha_k_new must be float or int. Found type {type(val)} at index {k}.")
        if val < 0:
            raise ValueError(f"Elements in alpha_k_new must be non-negative. Found {val} at index {k}.")

    if not isinstance(b_k_new, list):
        raise TypeError(f"b_k_new must be a list. Got {type(b_k_new)}.")
    if len(b_k_new) != num_new_call_classes:
        raise ValueError(f"Length of b_k_new ({len(b_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, val in enumerate(b_k_new):
        if not isinstance(val, int):
            raise TypeError(f"Elements in b_k_new must be integers. Found type {type(val)} at index {k}.")
        if val <= 0:
            raise ValueError(f"Elements in b_k_new must be positive. Found {val} at index {k}.")

    if not isinstance(B_cl_minus_1_new, list):
        raise TypeError(f"B_cl_minus_1_new must be a list. Got {type(B_cl_minus_1_new)}.")
    if len(B_cl_minus_1_new) != num_new_call_classes:
        raise ValueError(f"Length of B_cl_minus_1_new ({len(B_cl_minus_1_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, sub_list in enumerate(B_cl_minus_1_new):
        if not isinstance(sub_list, list):
            raise TypeError(f"Elements of B_cl_minus_1_new must be lists. Found type {type(sub_list)} at index {k}.")
        for l_idx, val in enumerate(sub_list):
            if not isinstance(val, (float, int)):
                raise TypeError(f"Elements in B_cl_minus_1_new[{k}] must be float or int. Found type {type(val)} at index {l_idx}.")
            if val < 0:
                raise ValueError(f"Elements in B_cl_minus_1_new[{k}] must be non-negative. Found {val} at index {l_idx}.")

    if not isinstance(t_k_new, list):
        raise TypeError(f"t_k_new must be a list. Got {type(t_k_new)}.")
    if len(t_k_new) != num_new_call_classes:
        raise ValueError(f"Length of t_k_new ({len(t_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, t_val in enumerate(t_k_new):
        if not isinstance(t_val, int):
            raise TypeError(f"Elements in t_k_new must be integers. Found type {type(t_val)} at index {k}.")
        if not (0 <= t_val <= C_total):
            raise ValueError(f"Elements in t_k_new must satisfy 0 <= t_val <= C_total. Found t_k_new[{k}] = {t_val} with C_total = {C_total}.")

    # 3. Consistency and Content for Handover Call Lists (alpha_k_ho, b_k_ho, B_cl_minus_1_ho, t_k_ho)
    if not isinstance(alpha_k_ho, list):
        raise TypeError(f"alpha_k_ho must be a list. Got {type(alpha_k_ho)}.")
    if len(alpha_k_ho) != num_ho_call_classes:
        raise ValueError(f"Length of alpha_k_ho ({len(alpha_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, val in enumerate(alpha_k_ho):
        if not isinstance(val, (float, int)):
            raise TypeError(f"Elements in alpha_k_ho must be float or int. Found type {type(val)} at index {k}.")
        if val < 0:
            raise ValueError(f"Elements in alpha_k_ho must be non-negative. Found {val} at index {k}.")
    
    if not isinstance(b_k_ho, list):
        raise TypeError(f"b_k_ho must be a list. Got {type(b_k_ho)}.")
    if len(b_k_ho) != num_ho_call_classes:
        raise ValueError(f"Length of b_k_ho ({len(b_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, val in enumerate(b_k_ho):
        if not isinstance(val, int):
            raise TypeError(f"Elements in b_k_ho must be integers. Found type {type(val)} at index {k}.")
        if val <= 0:
            raise ValueError(f"Elements in b_k_ho must be positive. Found {val} at index {k}.")

    if not isinstance(B_cl_minus_1_ho, list):
        raise TypeError(f"B_cl_minus_1_ho must be a list. Got {type(B_cl_minus_1_ho)}.")
    if len(B_cl_minus_1_ho) != num_ho_call_classes:
        raise ValueError(f"Length of B_cl_minus_1_ho ({len(B_cl_minus_1_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, sub_list in enumerate(B_cl_minus_1_ho):
        if not isinstance(sub_list, list):
            raise TypeError(f"Elements of B_cl_minus_1_ho must be lists. Found type {type(sub_list)} at index {k}.")
        for l_idx, val in enumerate(sub_list):
            if not isinstance(val, (float, int)):
                raise TypeError(f"Elements in B_cl_minus_1_ho[{k}] must be float or int. Found type {type(val)} at index {l_idx}.")
            if val < 0:
                raise ValueError(f"Elements in B_cl_minus_1_ho[{k}] must be non-negative. Found {val} at index {l_idx}.")

    if not isinstance(t_k_ho, list):
        raise TypeError(f"t_k_ho must be a list. Got {type(t_k_ho)}.")
    if len(t_k_ho) != num_ho_call_classes:
        raise ValueError(f"Length of t_k_ho ({len(t_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, t_val in enumerate(t_k_ho):
        if not isinstance(t_val, int):
            raise TypeError(f"Elements in t_k_ho must be integers. Found type {type(t_val)} at index {k}.")
        if not (0 <= t_val <= C_total): 
            raise ValueError(f"Elements in t_k_ho must satisfy 0 <= t_val <= C_total. Found t_k_ho[{k}] = {t_val} with C_total = {C_total}.")

    # --- INÍCIO DA LÓGICA DE CÁLCULO (APÓS VALIDAÇÕES) ---
    # 1. Inicialização de q_j_dist
    # Validações já garantem C_total >= 0
    q_j_dist = [0.0] * (C_total + 1) 
    q_j_dist[0] = 1.0
    # Se C_total = 0, q_j_dist = [1.0]

    # 2. Loop Recursivo Principal
    # O loop DEVE sempre ir de 1 até C_total.
    for j_iter in range(1, C_total + 1): # j_iter corresponde a 'j' no pseudocódigo
        total_contribution_for_j = 0.0

        # Loop para classes de NOVAS CHAMADAS
        for k_idx in range(num_new_call_classes):
            current_alpha_new = float(alpha_k_new[k_idx])
            current_b_new = b_k_new[k_idx]
            current_B_dist_new = B_cl_minus_1_new[k_idx]
            current_t_k_new = t_k_new[k_idx]

            # A condição IF decide se a classe contribui
            if j_iter <= (C_total - current_t_k_new):
                if current_alpha_new == 0.0:
                    continue # Pula para a próxima classe se alpha é zero

                max_l_new = floor(j_iter / current_b_new)
                if max_l_new < 1:
                    continue # Pula se nenhum lote puder compor j_iter

                soma_interna_new = 0.0
                for l_val in range(1, max_l_new + 1): # l_val corresponde a 'z' no pseudocódigo
                    B_list_idx = l_val - 1
                    B_prob_ge_l_new = 0.0
                    if B_list_idx < len(current_B_dist_new):
                        B_prob_ge_l_new = float(current_B_dist_new[B_list_idx])
                    
                    q_term_idx = j_iter - l_val * current_b_new
                    # q_term_idx >= 0 é garantido por max_l_new
                    # A validação q_term_idx < len(q_j_dist) também deve ser verdadeira
                    # Removendo verificação defensiva if q_term_idx < len(q_j_dist) conforme pseudocódigo implícito
                    soma_interna_new += q_j_dist[q_term_idx] * B_prob_ge_l_new
                
                total_contribution_for_j += current_alpha_new * current_b_new * soma_interna_new

        # Loop para classes de CHAMADAS DE HANDOVER
        for k_idx in range(num_ho_call_classes):
            current_alpha_ho = float(alpha_k_ho[k_idx])
            current_b_ho = b_k_ho[k_idx]
            current_B_dist_ho = B_cl_minus_1_ho[k_idx]
            current_t_k_ho = t_k_ho[k_idx]

            # A mesma lógica condicional aqui
            if j_iter <= (C_total - current_t_k_ho):
                if current_alpha_ho == 0.0:
                    continue

                max_l_ho = floor(j_iter / current_b_ho)
                if max_l_ho < 1:
                    continue

                soma_interna_ho = 0.0
                for l_val in range(1, max_l_ho + 1): # l_val corresponde a 'z'
                    B_list_idx = l_val - 1
                    B_prob_ge_l_ho = 0.0
                    if B_list_idx < len(current_B_dist_ho):
                        B_prob_ge_l_ho = float(current_B_dist_ho[B_list_idx])

                    q_term_idx = j_iter - l_val * current_b_ho
                    # Removendo verificação defensiva if q_term_idx < len(q_j_dist) conforme pseudocódigo implícito
                    soma_interna_ho += q_j_dist[q_term_idx] * B_prob_ge_l_ho
                
                total_contribution_for_j += current_alpha_ho * current_b_ho * soma_interna_ho

        # Fim dos loops de classes
        if j_iter > 0: # Sempre verdade neste loop
            q_j_dist[j_iter] = total_contribution_for_j / j_iter

    # 3. Normalização
    # Deve ser feita APÓS o loop principal estar completo.
    soma_total_q = sum(q_j_dist)
    if soma_total_q > 1e-9: # Usar uma pequena épsilon para comparação de ponto flutuante
        q_j_dist_normalizado = [val / soma_total_q for val in q_j_dist]
    else:
        # Este caso não deve ser alcançado se q_j_dist[0]=1.0 e C_total >= 0
        # e pelo menos alguma carga for diferente de zero com t_k permissivos.
        # Se todas as cargas forem zero, ou todos t_k forem muito restritivos,
        # q_j_dist será [1.0, 0, 0, ...], e soma_total_q = 1.0.
        raise RuntimeError(f"Normalization failed: sum of q_j_dist is not positive ({soma_total_q}). q_j_dist={q_j_dist}")

    return q_j_dist_normalizado
    # --- FIM DA LÓGICA DE CÁLCULO ---
    if C_total < 0:
        raise ValueError(f"C_total must be non-negative. Got {C_total}.")

    if not isinstance(num_new_call_classes, int):
        raise TypeError(f"num_new_call_classes must be an integer. Got {type(num_new_call_classes)}.")
    if num_new_call_classes < 0:
        raise ValueError(f"num_new_call_classes must be non-negative. Got {num_new_call_classes}.")

    if not isinstance(num_ho_call_classes, int):
        raise TypeError(f"num_ho_call_classes must be an integer. Got {type(num_ho_call_classes)}.")
    if num_ho_call_classes < 0:
        raise ValueError(f"num_ho_call_classes must be non-negative. Got {num_ho_call_classes}.")

    # 2. Consistency and Content for New Call Lists (alpha, b, B_cl)
    if not isinstance(alpha_k_new, list):
        raise TypeError(f"alpha_k_new must be a list. Got {type(alpha_k_new)}.")
    if len(alpha_k_new) != num_new_call_classes:
        raise ValueError(f"Length of alpha_k_new ({len(alpha_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, val in enumerate(alpha_k_new):
        if not isinstance(val, (float, int)):
            raise TypeError(f"Elements in alpha_k_new must be float or int. Found type {type(val)} at index {k}.")
        if val < 0:
            raise ValueError(f"Elements in alpha_k_new must be non-negative. Found {val} at index {k}.")

    if not isinstance(b_k_new, list):
        raise TypeError(f"b_k_new must be a list. Got {type(b_k_new)}.")
    if len(b_k_new) != num_new_call_classes:
        raise ValueError(f"Length of b_k_new ({len(b_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, val in enumerate(b_k_new):
        if not isinstance(val, int):
            raise TypeError(f"Elements in b_k_new must be integers. Found type {type(val)} at index {k}.")
        if val <= 0:
            raise ValueError(f"Elements in b_k_new must be positive. Found {val} at index {k}.")

    if not isinstance(B_cl_minus_1_new, list):
        raise TypeError(f"B_cl_minus_1_new must be a list. Got {type(B_cl_minus_1_new)}.")
    if len(B_cl_minus_1_new) != num_new_call_classes:
        raise ValueError(f"Length of B_cl_minus_1_new ({len(B_cl_minus_1_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, sub_list in enumerate(B_cl_minus_1_new):
        if not isinstance(sub_list, list):
            raise TypeError(f"Elements of B_cl_minus_1_new must be lists. Found type {type(sub_list)} at index {k}.")
        for l_idx, val in enumerate(sub_list):
            if not isinstance(val, (float, int)):
                raise TypeError(f"Elements in B_cl_minus_1_new[{k}] must be float or int. Found type {type(val)} at index {l_idx}.")
            if val < 0:
                raise ValueError(f"Elements in B_cl_minus_1_new[{k}] must be non-negative. Found {val} at index {l_idx}.")

    # 3. Consistency and Content for Handover Call Lists (alpha, b, B_cl)
    if not isinstance(alpha_k_ho, list):
        raise TypeError(f"alpha_k_ho must be a list. Got {type(alpha_k_ho)}.")
    if len(alpha_k_ho) != num_ho_call_classes:
        raise ValueError(f"Length of alpha_k_ho ({len(alpha_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, val in enumerate(alpha_k_ho):
        if not isinstance(val, (float, int)):
            raise TypeError(f"Elements in alpha_k_ho must be float or int. Found type {type(val)} at index {k}.")
        if val < 0:
            raise ValueError(f"Elements in alpha_k_ho must be non-negative. Found {val} at index {k}.")
    
    if not isinstance(b_k_ho, list):
        raise TypeError(f"b_k_ho must be a list. Got {type(b_k_ho)}.")
    if len(b_k_ho) != num_ho_call_classes:
        raise ValueError(f"Length of b_k_ho ({len(b_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, val in enumerate(b_k_ho):
        if not isinstance(val, int):
            raise TypeError(f"Elements in b_k_ho must be integers. Found type {type(val)} at index {k}.")
        if val <= 0:
            raise ValueError(f"Elements in b_k_ho must be positive. Found {val} at index {k}.")

    if not isinstance(B_cl_minus_1_ho, list):
        raise TypeError(f"B_cl_minus_1_ho must be a list. Got {type(B_cl_minus_1_ho)}.")
    if len(B_cl_minus_1_ho) != num_ho_call_classes:
        raise ValueError(f"Length of B_cl_minus_1_ho ({len(B_cl_minus_1_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, sub_list in enumerate(B_cl_minus_1_ho):
        if not isinstance(sub_list, list):
            raise TypeError(f"Elements of B_cl_minus_1_ho must be lists. Found type {type(sub_list)} at index {k}.")
        for l_idx, val in enumerate(sub_list):
            if not isinstance(val, (float, int)):
                raise TypeError(f"Elements in B_cl_minus_1_ho[{k}] must be float or int. Found type {type(val)} at index {l_idx}.")
            if val < 0:
                raise ValueError(f"Elements in B_cl_minus_1_ho[{k}] must be non-negative. Found {val} at index {l_idx}.")

    # 4. Validações para t_k_new
    if not isinstance(t_k_new, list):
        raise TypeError(f"t_k_new must be a list. Got {type(t_k_new)}.")
    if len(t_k_new) != num_new_call_classes:
        raise ValueError(f"Length of t_k_new ({len(t_k_new)}) must match num_new_call_classes ({num_new_call_classes}).")
    for k, t_val in enumerate(t_k_new):
        if not isinstance(t_val, int):
            raise TypeError(f"Elements in t_k_new must be integers. Found type {type(t_val)} at index {k}.")
        # Docstring: 0 <= t_k < C_total. If C_total is 0, then t_k must be 0.
        # A class is blocked if C_total - t_val == 0.
        # If t_val = C_total, then C_total - t_val = 0, so class k is blocked if j > 0.
        # If t_val = 0, then C_total - t_val = C_total, class k can use all channels.
        if not (0 <= t_val <= C_total):
            raise ValueError(f"Elements in t_k_new must satisfy 0 <= t_val <= C_total. Found t_k_new[{k}] = {t_val} with C_total = {C_total}.")

    # 5. Validações para t_k_ho
    if not isinstance(t_k_ho, list):
        raise TypeError(f"t_k_ho must be a list. Got {type(t_k_ho)}.")
    if len(t_k_ho) != num_ho_call_classes:
        raise ValueError(f"Length of t_k_ho ({len(t_k_ho)}) must match num_ho_call_classes ({num_ho_call_classes}).")
    for k, t_val in enumerate(t_k_ho):
        if not isinstance(t_val, int):
            raise TypeError(f"Elements in t_k_ho must be integers. Found type {type(t_val)} at index {k}.")
        if not (0 <= t_val <= C_total): # Same logic as for t_k_new
            raise ValueError(f"Elements in t_k_ho must satisfy 0 <= t_val <= C_total. Found t_k_ho[{k}] = {t_val} with C_total = {C_total}.")
            
    # --- INÍCIO DA LÓGICA DE CÁLCULO (APÓS VALIDAÇÕES) ---
    # 1. Inicialização
    q_j_dist = [0.0] * (C_total + 1) # Usar q_j_dist como nome da variável
    if C_total >= 0: # Validação já garante C_total >= 0
        q_j_dist[0] = 1.0
    # Se C_total = -1 (impossível devido à validação), q_j_dist seria lista vazia ou erro.
    # Se C_total = 0, q_j_dist = [1.0]

    # 2. Loop Recursivo Principal
    # O loop DEVE sempre ir de 1 até C_total.
    for j_iter in range(1, C_total + 1): # j_iter corresponde a 'j' no pseudocódigo
        total_contribution_for_j = 0.0

        # Loop para classes de NOVAS CHAMADAS
        for k_idx in range(num_new_call_classes):
            current_alpha_new = float(alpha_k_new[k_idx])
            current_b_new = b_k_new[k_idx]
            current_B_dist_new = B_cl_minus_1_new[k_idx]
            current_t_k_new = t_k_new[k_idx]

            # A condição IF decide se a classe contribui, mas NÃO para o loop principal.
            if j_iter <= (C_total - current_t_k_new):
                if current_alpha_new == 0.0:
                    continue # Pula para a próxima classe se alpha é zero

                max_l_new = floor(j_iter / current_b_new)
                if max_l_new < 1:
                    continue # Pula se nenhum lote puder compor j_iter

                soma_interna_new = 0.0
                for l_val in range(1, max_l_new + 1): # l_val corresponde a 'z' no pseudocódigo
                    B_list_idx = l_val - 1
                    B_prob_ge_l_new = 0.0
                    if B_list_idx < len(current_B_dist_new):
                        B_prob_ge_l_new = float(current_B_dist_new[B_list_idx])
                    
                    q_term_idx = j_iter - l_val * current_b_new
                    # q_term_idx >= 0 é garantido por max_l_new
                    # A validação q_term_idx < len(q_j_dist) também deve ser verdadeira
                    # Removendo verificação defensiva if q_term_idx < len(q_j_dist) conforme pseudocódigo implícito
                    soma_interna_new += q_j_dist[q_term_idx] * B_prob_ge_l_new
                
                total_contribution_for_j += current_alpha_new * current_b_new * soma_interna_new

        # Loop para classes de CHAMADAS DE HANDOVER
        for k_idx in range(num_ho_call_classes):
            current_alpha_ho = float(alpha_k_ho[k_idx])
            current_b_ho = b_k_ho[k_idx]
            current_B_dist_ho = B_cl_minus_1_ho[k_idx]
            current_t_k_ho = t_k_ho[k_idx]

            # A mesma lógica condicional aqui
            if j_iter <= (C_total - current_t_k_ho):
                if current_alpha_ho == 0.0:
                    continue

                max_l_ho = floor(j_iter / current_b_ho)
                if max_l_ho < 1:
                    continue

                soma_interna_ho = 0.0
                for l_val in range(1, max_l_ho + 1): # l_val corresponde a 'z'
                    B_list_idx = l_val - 1
                    B_prob_ge_l_ho = 0.0
                    if B_list_idx < len(current_B_dist_ho):
                        B_prob_ge_l_ho = float(current_B_dist_ho[B_list_idx])

                    q_term_idx = j_iter - l_val * current_b_ho
                    # Removendo verificação defensiva if q_term_idx < len(q_j_dist) conforme pseudocódigo implícito
                    soma_interna_ho += q_j_dist[q_term_idx] * B_prob_ge_l_ho
                
                total_contribution_for_j += current_alpha_ho * current_b_ho * soma_interna_ho

        # Fim dos loops de classes
        if j_iter > 0: # Sempre verdade neste loop
            q_j_dist[j_iter] = total_contribution_for_j / j_iter

    # 3. Normalização
    # Deve ser feita APÓS o loop principal estar completo.
    soma_total_q = sum(q_j_dist)
    q_j_dist_normalizado: List[float]
    if soma_total_q > 1e-9: # Usar uma pequena épsilon para comparação de ponto flutuante
        q_j_dist_normalizado = [val / soma_total_q for val in q_j_dist]
    else:
        # Este caso não deve ser alcançado se q_j_dist[0]=1.0 e C_total >= 0
        raise RuntimeError(f"Normalization failed: sum of q_j_dist is not positive ({soma_total_q}). q_j_dist={q_j_dist}")
    
    return q_j_dist_normalizado
    # --- FIM DA LÓGICA DE CÁLCULO ---
    C_total_val_fail = 2
    q_j_dist_val_fail = [0.2, 0.3, 0.5] # Soma = 1.0
    b_k_new_val_fail = [1, 3] # b_k=3 > C_total_val_fail=2
    b_k_ho_val_fail = []
    print(f"  Parâmetros: C_total={C_total_val_fail}, q_j_dist (soma)={sum(q_j_dist_val_fail):.1f}, b_k_new={b_k_new_val_fail}, b_k_ho={b_k_ho_val_fail}")
    try:
        calculate_blocking_probabilities_cs(q_j_dist_val_fail, C_total_val_fail, b_k_new_val_fail, b_k_ho_val_fail)
        print("  Falha no teste de validação: ValueError não foi levantado.")
    except ValueError as e:
        print(f"  Teste de validação falhou como esperado: {e}")
    except Exception as e:
        print(f"  Falha no teste de validação: Exceção inesperada: {e}")
    print("-" * 50)

    # 3. Teste de Validação (q_j_dist não normalizada)
    print("\n--- Teste de Validação (q_j_dist não normalizada) para calculate_blocking_probabilities_cs ---")
    C_total_val_norm_fail = 2
    q_j_dist_val_norm_fail = [0.2, 0.3, 0.6] # Soma = 1.1
    b_k_new_val_norm_fail = [1]
    b_k_ho_val_norm_fail = []
    print(f"  Parâmetros: C_total={C_total_val_norm_fail}, q_j_dist (soma)={sum(q_j_dist_val_norm_fail):.1f}, b_k_new={b_k_new_val_norm_fail}, b_k_ho={b_k_ho_val_norm_fail}")
    try:
        calculate_blocking_probabilities_cs(q_j_dist_val_norm_fail, C_total_val_norm_fail, b_k_new_val_norm_fail, b_k_ho_val_norm_fail)
        print("  Falha no teste de validação: ValueError (q_j_dist não normalizada) não foi levantado.")
    except ValueError as e:
        print(f"  Teste de validação falhou como esperado: {e}")
    except Exception as e:
        print(f"  Falha no teste de validação: Exceção inesperada: {e}")
    print("-" * 50)
