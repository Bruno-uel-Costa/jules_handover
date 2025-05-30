# Content of helper.py
import numpy as np
import scipy.interpolate as si
import torch
import torch.nn as nn # Added for nn.Parameter, nn.init
from torch.nn.parameter import Parameter


def get_knots(start, end, n_bases=5, spline_order=3):
    """
    Arguments:
        x; torch.tensor of dim 1
    """
    x_range = end - start
    # Add a small epsilon if start and end are the same, to prevent division by zero in dknots
    if x_range == 0:
        x_range = torch.tensor(1.0) # Or some small value
        # Adjust start and end slightly to create a range
        # This case should ideally be handled by the user ensuring grid_range[0] < grid_range[1]
        # For now, let's assume start < end from grid_range.
        # If they can be equal, a more robust handling for dknots is needed.

    start_calc = start - x_range * 0.001 # Use different var names
    end_calc = end + x_range * 0.001
    
    m = spline_order - 1 # degree
    # nk = n_bases - m  # number of interior knots, this is from "Generalized Additive Models" by Wood (2017) for P-splines
                        # For B-splines, n_bases = number of knots - spline_order - 1 for scipy's convention if knots are non-repeating at ends.
                        # Or, more simply, number of basis functions is often set directly.
                        # The original helper.py logic for knots seems to aim for n_bases functions.
                        # n_knots = n_bases + spline_order + 1 (common for some definitions)
                        # Let's stick to original helper.py knot calculation logic for now.
    
    # Original logic from helper.py: nk = n_bases - m for number of *interior* knots.
    # Total knots = nk + 2 * (m+1) for padding.
    # Let's re-verify the knot calculation logic based on n_bases directly.
    # Typically, for `n_bases` basis functions of order `k` (spline_order = k-1),
    # you need `n_bases + k` knots if using open uniform knot vectors.
    # Or, `n_bases + 2*spline_order` knots if boundary knots are repeated `spline_order` times.

    # The get_knots in helper.py seems to be:
    # m = spline_order - 1 (this is `degree k` if spline_order is `k+1` as in scipy. Usually spline_order `k` means degree `k-1`)
    # Let's assume spline_order is `k` (degree).
    # Scipy uses `k` for degree. So if spline_order=3 (cubic), degree is 3.
    # Number of knots `n_knots = n_bases + k + 1`.
    # The original code:
    # m = spline_order - 1. If spline_order = 3 (cubic), m = 2.
    # nk = n_bases - m. If n_bases = 5, nk = 3 (interior knots).
    # dknots = (end_calc - start_calc) / (nk -1) if nk > 1. If nk=1, dknots is not well-defined here.
    # If n_bases <= m (spline_order-1), then nk <=0. This needs careful handling.
    # Example: n_bases=3, spline_order=3 -> m=2, nk=1. (end_calc - start_calc) / 0 -> NaN/Inf
    
    if n_bases <= spline_order -1 : # nk <= 0, problematic for dknots
        # Fallback: minimal knots for this case or raise error
        # For a single basis function (n_bases=1), you still need spline_order+1 knots.
        # This part needs to be robust. Let's assume n_bases > spline_order - 1 for now
        # or use a simpler knot generation if n_bases is small.
        # For KAN, grid_size (n_bases) is usually small like 5, spline_order 3. So 5 > 2, nk=3. This is fine.
        if n_bases < 2 : # Must have at least 2 bases for this knot calculation.
             # Fallback for very few bases (e.g. linear spline with n_bases=2)
             # This is a simplification, robust knot placement is complex.
            knots = torch.linspace(start_calc, end_calc, steps=n_bases + 2 * (spline_order-1) + 2) # Adjusted steps
        else: # Original logic was problematic for nk=1 (n_bases = spline_order)
            num_internal_knots = n_bases - (spline_order -1) # nk from original
            if num_internal_knots <=1: # Handles n_bases = spline_order or n_bases = spline_order -1
                # Simplified knot spacing for few internal knots
                # This ensures at least two points for linspace if num_internal_knots=0 or 1
                # leading to (end_calc - start_calc) / (some_positive_number)
                # This is a deviation from original if nk=1 (e.g. n_bases=3, spline_order=3)
                # Original: dknots = (end_calc - start_calc) / 0
                # Let's use a method that's robust for small num_internal_knots
                 num_segments = num_internal_knots +1 if num_internal_knots > 0 else 1
                 dknots = (end_calc - start_calc) / num_segments

            else: # num_internal_knots > 1
                dknots = (end_calc - start_calc) / (num_internal_knots -1)

            # knots for padding: m+1 on each side. m = spline_order - 1
            # total steps = num_internal_knots + 2 * (spline_order -1 + 1) = num_internal_knots + 2 * spline_order
            knots = torch.linspace(
                start=start_calc - dknots * spline_order, # Adjusted padding to use spline_order directly
                end=end_calc + dknots * spline_order,
                steps=num_internal_knots + 2 * spline_order
            )
    else: # n_bases > spline_order -1
        m = spline_order - 1 
        nk = n_bases - m # number of interior knots
        dknots = (end_calc - start_calc) / (nk - 1)
        knots = torch.linspace(
            start=start_calc - dknots * (m + 1), 
            end=end_calc + dknots * (m + 1), 
            steps=nk + 2 * (m + 1) # Original had m+2, typo? should be 2*m+2 or 2*(m+1)
                                    # nk + 2m + 2
        )
    return knots.float()


def get_X_spline(x, knots, n_bases=5, spline_order=3, add_intercept=True):
    cuda = x.is_cuda
    if len(x.shape) != 1:
        raise ValueError("x has to be 1 dimentional for get_X_spline")
    
    # Ensure knots and other params for tck are on the same device as x if possible, or CPU for scipy
    knots_np = knots.cpu().numpy()
    
    # tck for scipy: knots, coefficients (c), degree (k)
    # Scipy's BSpline uses degree k (spline_order).
    # n_bases here is the number of basis functions.
    # Number of data points is len(x).
    # X will be (len(x), n_bases)
    
    X_spl = torch.zeros([len(x), n_bases], dtype=x.dtype, device=x.device)
    x_np = x.cpu().numpy()

    for i in range(n_bases):
        # Create a coefficient vector for the i-th basis function
        # This is a standard way to evaluate individual basis functions with splev
        vec_c = np.zeros(n_bases)
        vec_c[i] = 1.0 
        
        # Construct tck for the i-th basis function.
        # The number of knots must be n_coeffs + degree + 1 for scipy's splev if c is given.
        # Here, we are providing coefficients for *basis functions*, not for spline interpolation of data.
        # Scipy's BSpline.basis_element or constructing BSpline objects is more direct for basis values.
        # The original approach uses splev with specific coefficient vectors.
        # For this to work, `knots` must be the full knot vector for the basis set,
        # and `n_bases` must be consistent with `len(knots) - spline_order - 1`.

        # Let's assume `knots` is correctly formed by `get_knots` such that `n_bases` basis functions can be derived.
        # The number of coefficients `c` should be `len(knots) - (degree+1)`.
        # If `n_bases` is this value, then `vec_c` is of the correct length.
        
        # Adjust `vec_c` length if it doesn't match `len(knots_np) - spline_order - 1`
        # This is a bit of a hack; implies `n_bases` might not be what `splev` expects for `c` length.
        # A common convention: n_coeffs = n_knots - degree - 1. If n_bases is n_coeffs, this is fine.
        # Let's trust the original structure from helper.py where n_bases is used for vec_c length.
        
        current_tck = (knots_np, vec_c, spline_order-1) # Scipy uses degree (k-1 if k is order)
                                                        # If spline_order is "order k", then degree is k.
                                                        # The original code had spline_order, which means degree for splev.
                                                        # Let's assume spline_order means degree for splev.
        
        # If spline_order is order (e.g. 3 for quadratic, 4 for cubic), then degree is spline_order-1
        # If spline_order is degree (e.g. 3 for cubic), then degree is spline_order.
        # The helper.py uses `m = spline_order - 1` as degree-like.
        # And BSpline class init has `spline_order` which seems to be degree.
        # Let's assume `spline_order` parameter is "degree k".
        
        # SciPy BSpline: k is degree. knots length M. coeffs length M-k-1.
        # So, if spline_order is degree k, len(c) = len(knots) - k - 1.
        # And n_bases should be this len(c).
        
        # The tck for splev: (knots, c, k=degree)
        # It seems the original code's `n_bases` corresponds to `len(c)`.
        # And `spline_order` to `k` (degree).
        
        tck_for_splev = (knots_np, vec_c, spline_order) # Assuming spline_order is degree
        
        basis_eval_np = si.splev(x_np, tck_for_splev, der=0)
        X_spl[:, i] = torch.from_numpy(basis_eval_np).to(x.device, dtype=x.dtype)

    if add_intercept: # This part seems incorrect for KAN, typically intercept handled by base_weights or a bias term
        ones = torch.ones_like(X_spl[:, :1])
        X_spl = torch.cat([ones, X_spl], dim=1) # Use torch.cat
    return X_spl


def get_S(n_bases=5, spline_order=3, add_intercept=True):
    S_np = np.identity(n_bases)
    # spline_order is degree k. For penalty, often use m-th difference, m=2 for cubic.
    # Original used m2 = spline_order - 1. If spline_order=3 (cubic degree), m2=2. This is typical.
    # If spline_order is "order" (k+1), then m2 = k.
    # Let's assume spline_order is degree.
    m2 = spline_order -1 # If spline_order = 3 (cubic), m2=2 (2nd order diff penalty)
    if m2 < 0: m2 = 0 # No negative differences

    for _ in range(m2): # Corrected loop to iterate m2 times
        S_np = np.diff(S_np, axis=0)
    
    if S_np.shape[0] == 0: # Handle case where diff results in empty matrix (e.g. n_bases=1, m2=1)
        S_np = np.zeros((n_bases, n_bases)) # Or some other appropriate small penalty
    else:
        S_np = np.dot(S_np.T, S_np)
        S_np = (S_np + S_np.T) / 2
    
    if add_intercept:
        zeros_h = np.zeros_like(S_np[:1, :])
        S_np = np.vstack([zeros_h, S_np])
        zeros_v = np.zeros_like(S_np[:, :1])
        S_np = np.hstack([zeros_v, S_np])
    return torch.from_numpy(S_np.astype(np.float32))


def _trunc(x, minval=None, maxval=None):
    x_clone = torch.clone(x)
    if minval is not None:
        x_clone[x_clone < minval] = minval
    if maxval is not None:
        x_clone[x_clone > maxval] = maxval
    return x_clone


# encodeSplines and corr2d_stack are not directly used by KANLayer as per current plan, can be kept or removed.
# For now, keeping them.
def encodeSplines(x, n_bases=5, spline_order=3, start=None, end=None, warn=True):
    if len(x.shape) == 1:
        x_reshaped = x.reshape((-1, 1)) 
    else:
        x_reshaped = x

    start_val, end_val = start, end
    if start_val is None: start_val = torch.amin(x_reshaped)
    if end_val is None: end_val = torch.amax(x_reshaped)

    if x_reshaped.min() < start_val:
        if warn: print("WARNING, x.min() < start. Truncating.")
        x_reshaped = _trunc(x_reshaped, minval=start_val)
    if x_reshaped.max() > end_val:
        if warn: print("WARNING, x.max() > end. Truncating.")
        x_reshaped = _trunc(x_reshaped, maxval=end_val)
    
    bs = BSpline(start_val.item(), end_val.item(), n_bases=n_bases, spline_order=spline_order) 

    n_rows, n_cols = x_reshaped.shape
    x_long = x_reshaped.reshape((-1,))
    x_feat = bs.predict(x_long, add_intercept=False) # shape (n_rows * n_cols, n_bases)
    return x_feat.reshape((n_rows, n_cols, n_bases))


def corr2d_stack(X_tensor, K_tensor):
    out = torch.stack([torch.matmul(x_item, k_item) for x_item, k_item in zip(X_tensor, K_tensor)]).squeeze(-1) 
    out = out.permute((1, 2, 0))
    return out

# End of helper.py content (with minor fixes)

class BSpline:
    def __init__(self, start=0.0, end=1.0, n_bases=10, spline_order=3): # Ensure float for start/end
        self.start = float(start)
        self.end = float(end)
        self.n_bases = n_bases
        self.spline_order = spline_order # This is degree k for splines
        
        # Ensure start < end for get_knots
        if self.start >= self.end:
            # Default to a small range if start >= end to prevent errors in knot calculation
            # A warning should be issued or this handled by KANLayer's grid_range validation
            # Forcing end > start for now for robustness in BSpline itself
            self.end = self.start + 1.0 
            # print(f"Warning: BSpline start >= end ({start}>={end}). Adjusted end to {self.end}")


        self.knots = get_knots(torch.tensor(self.start), torch.tensor(self.end), self.n_bases, self.spline_order)
        
        # Store S as numpy array, convert to tensor on demand
        s_matrix_np = get_S(self.n_bases, self.spline_order, add_intercept=False).numpy()
        # Ensure S matrix has correct dimensions even if n_bases or spline_order is small
        if s_matrix_np.shape != (self.n_bases, self.n_bases):
             s_matrix_np = np.zeros((self.n_bases, self.n_bases), dtype=np.float32) # Fallback
        self.S_np = s_matrix_np


    def __repr__(self):
        return f"BSpline(start={self.start}, end={self.end}, n_bases={self.n_bases}, spline_order={self.spline_order})"

    def getS_tensor(self, add_intercept=False):
        S_np_local = self.S_np
        if add_intercept:
            # This logic for adding intercept to S should ensure correct dimensions
            if S_np_local.shape[0] > 0 and S_np_local.shape[1] > 0 : # only if S is not empty
                zeros_h = np.zeros((1, S_np_local.shape[1]), dtype=S_np_local.dtype)
                S_np_local = np.vstack([zeros_h, S_np_local])
                zeros_v = np.zeros((S_np_local.shape[0], 1), dtype=S_np_local.dtype)
                S_np_local = np.hstack([zeros_v, S_np_local])
            else: # if S was empty or 0x0, create appropriate sized matrix if adding intercept
                target_dim = self.n_bases + 1 if self.n_bases > 0 else 1
                S_np_local = np.zeros((target_dim, target_dim), dtype=np.float32)

        return torch.from_numpy(S_np_local.astype(np.float32))

    def predict(self, x, add_intercept=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        current_device = x.device
        
        # Clamp x to be within [start, end] of the spline definition
        # using tensor operations for clamping
        x_clamped = torch.clamp(x, min=self.start, max=self.end)
        
        knots_device = self.knots.to(current_device)

        # get_X_spline expects add_intercept=False for KANLayer use typically
        # (intercept/base handled by base_weights)
        return get_X_spline(
            x=x_clamped, # Use clamped x
            knots=knots_device,
            n_bases=self.n_bases,
            spline_order=self.spline_order,
            add_intercept=add_intercept 
        )

# KANLayer Implementation
class KANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3, grid_range=[-1.0, 1.0]):
        super(KANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size # Number of basis functions for each spline
        self.spline_order = spline_order # Degree of B-splines
        self.grid_range = grid_range

        if grid_range[0] >= grid_range[1]:
            raise ValueError(f"grid_range[0] must be less than grid_range[1]. Got: {grid_range}")

        # Spline Coefficients: one set for each (output_dim, input_dim) pair
        # Shape: (output_dim, input_dim, grid_size)
        self.spline_coeffs = nn.Parameter(torch.empty(output_dim, input_dim, grid_size))
        nn.init.normal_(self.spline_coeffs, mean=0.0, std=0.1) # Example initialization

        # Base Weights: one for each (output_dim, input_dim) pair
        # These act like linear weights for the raw input features
        self.base_weights = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.normal_(self.base_weights, mean=0.0, std=0.1) # Example initialization

        # Create B-spline evaluators for each input dimension
        # These are not nn.Modules themselves but are used to compute basis values
        self.b_spline_evaluators = []
        for _ in range(input_dim):
            spline_evaluator = BSpline(
                start=grid_range[0], 
                end=grid_range[1], 
                n_bases=grid_size, 
                spline_order=spline_order
            )
            self.b_spline_evaluators.append(spline_evaluator)

    def forward(self, x):
        # Input x shape: (batch_size, input_dim)
        if x.shape[1] != self.input_dim:
            raise ValueError(f"Input feature dimension mismatch. Expected {self.input_dim}, got {x.shape[1]}")

        batch_size = x.shape[0]
        
        # Initialize output tensor
        # spline_output will store sum_i phi_ij(x_i) for each j
        total_output = torch.zeros(batch_size, self.output_dim, device=x.device, dtype=x.dtype)

        for j in range(self.output_dim): # Iterate over output dimensions
            for i in range(self.input_dim): # Iterate over input dimensions (features)
                input_feature_i = x[:, i] # Shape: (batch_size)

                # 1. Base term: w_ij * x_i
                base_contribution_ij = self.base_weights[j, i] * input_feature_i
                
                # 2. Spline term: sum_k (c_ijk * B_k(x_i))
                # BSpline.predict expects a 1D tensor (batch of values for one feature)
                # and returns (batch_size, grid_size)
                basis_values_i = self.b_spline_evaluators[i].predict(input_feature_i, add_intercept=False)
                basis_values_i = basis_values_i.to(x.device, dtype=x.dtype) # Ensure device and dtype

                current_spline_coeffs_ji = self.spline_coeffs[j, i, :] # Shape: (grid_size)
                
                # Einstein summation: 'bk,k->b' means sum over k for each b in batch
                # basis_values_i: (batch_size, grid_size)
                # current_spline_coeffs_ji: (grid_size)
                # spline_activation_ij: (batch_size)
                spline_activation_ij = torch.einsum('bk,k->b', basis_values_i, current_spline_coeffs_ji)
                
                # Add contributions for this phi_ij(x_i) to the j-th output
                total_output[:, j] += base_contribution_ij + spline_activation_ij
                
        return total_output


if __name__ == '__main__':
    print("kan_layer.py executed as main.")
    
    # Test BSpline (basic check, more thorough tests for knots and eval if issues arise)
    print("\n--- BSpline Test (from KANLayer context) ---")
    bs_test = BSpline(start=0.0, end=1.0, n_bases=5, spline_order=3)
    print(bs_test)
    test_x_bs = torch.linspace(0.0, 1.0, 10, dtype=torch.float32)
    print("Test x for BSpline:", test_x_bs)
    bs_eval = bs_test.predict(test_x_bs)
    print("BSpline evaluation shape:", bs_eval.shape) # Expected: (10, 5)
    # print("BSpline evaluation values (first 2):", bs_eval[:2])
    S_matrix = bs_test.getS_tensor(add_intercept=False)
    print("S matrix shape:", S_matrix.shape) # Expected: (5,5)

    # Test KANLayer
    print("\n--- KANLayer Test ---")
    input_d, output_d = 3, 2
    batch_s = 4
    grid_s = 5
    spline_o = 3
    grid_r = [0.0, 1.0] # Assuming inputs are normalized to [0,1] for this test

    kan_layer = KANLayer(input_d, output_d, grid_size=grid_s, spline_order=spline_o, grid_range=grid_r)
    print(f"KANLayer created: input_dim={input_d}, output_dim={output_d}, grid_size={grid_s}, spline_order={spline_o}, grid_range={grid_r}")

    # Create dummy input normalized to grid_range for meaningful spline activation
    dummy_kan_input = torch.rand(batch_s, input_d, dtype=torch.float32) # Values in [0,1]
    # If grid_range was [-1,1], input would be: 2 * torch.rand(batch_s, input_d) - 1
    
    print("KANLayer input shape:", dummy_kan_input.shape)
    print("KANLayer input (first sample):", dummy_kan_input[0])

    # Perform a forward pass
    kan_output = kan_layer(dummy_kan_input)
    print("KANLayer output shape:", kan_output.shape) # Expected: (batch_s, output_d) i.e. (4,2)
    print("KANLayer output (first sample):", kan_output[0])

    # Check if gradients can be computed (requires_grad should be True for parameters)
    print(f"\nKANLayer spline_coeffs requires_grad: {kan_layer.spline_coeffs.requires_grad}")
    print(f"KANLayer base_weights requires_grad: {kan_layer.base_weights.requires_grad}")
    
    # Simple loss and backward pass test
    if kan_output.requires_grad: # Output should require grad if input does or params do
        try:
            dummy_target = torch.randn(batch_s, output_d, dtype=torch.float32)
            loss = nn.MSELoss()(kan_output, dummy_target)
            loss.backward()
            print("Backward pass successful.")
            # Check if gradients exist for a parameter
            if kan_layer.spline_coeffs.grad is not None:
                print("Gradients computed for spline_coeffs (sum):", torch.sum(kan_layer.spline_coeffs.grad).item())
            else:
                print("No gradients for spline_coeffs after backward.")
            if kan_layer.base_weights.grad is not None:
                print("Gradients computed for base_weights (sum):", torch.sum(kan_layer.base_weights.grad).item())
            else:
                print("No gradients for base_weights after backward.")

        except Exception as e:
            print(f"Error during backward pass test: {e}")
    else:
        # If input requires_grad=False (like torch.rand default), output might not require grad
        # unless a parameter itself caused it. For a layer with learnable weights, output should require grad.
        print("KANLayer output does not require grad. Check input.requires_grad or layer parameters.")


    print("\n--- Tests Finished ---")
    pass
