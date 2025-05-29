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

    # 2. Teste de Validação (Falha Esperada)
    print("\n--- Teste de Validação (Falha Esperada) para calculate_blocking_probabilities_cs ---")
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
