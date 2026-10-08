import torch
import torch.nn as nn
import torch.nn.functional as F


class Autocast:

    def __init__(self, enabled=True, dtype=torch.float16):
        self.enabled = enabled and torch.cuda.is_available()
        self.target_dtype = dtype
        self._original_funcs = {}
        self._depth = 0

        # Define the functions we want to intercept and cast.
        # Format: (module, function_name_string)
        self.DOWNCAST_OPS = [
            (torch, 'matmul'),
            (torch, 'bmm'),
            (F, 'linear'),
        ]
        self.UPCAST_OPS = [
            (F, 'layer_norm'),
            (F, 'cross_entropy'),
        ]


    def _create_wrapper(self, original_func, target_dtype):
        """
        Creates a wrapper that casts inputs to the target dtype
        and calls the function with casted arguments
        """
        def wrapper(*args, **kwargs):
            """
            Wrapper that casts all inputs to the target dtype 
            and calls the function with casted arguments
            """

            # Cast all tensor arguments in args and kwargs
            new_args = []
            for arg in args:
                if torch.is_tensor(arg):
                    if torch.is_floating_point(arg):
                        new_args.append(arg.to(dtype=target_dtype))
                        continue
                new_args.append(arg)
            new_kwargs = {}
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    if torch.is_floating_point(value):
                        new_kwargs[key] = value.to(dtype=target_dtype)
                        continue
                new_kwargs[key] = value

            # Call the original function with potentially casted inputs
            return original_func(*new_args, **new_kwargs)

        return wrapper

    def __enter__(self):
        """
        Wraps all the functions from DOWNCAST_OPS and UPCAST_OPS,
        Stores original functions
        And sets wrapped functions instead of original ones in the module
        """
        if not self.enabled:
            return self

        self._depth += 1
        if self._depth > 1:
            return self

        # Store original functions and apply patches
        for module, func_name in self.DOWNCAST_OPS + self.UPCAST_OPS:
            # Store original function
            original_func = getattr(module, func_name)
            self._original_funcs[(module, func_name)] = original_func

            # Create wrapped version of the function
            # Note that you need different target_dtype for DOWNCAST_OPS and UPCAST_OPS
            if (module, func_name) in self.DOWNCAST_OPS:
                wrapped_func = self._create_wrapper(original_func, self.target_dtype)
            else:
                wrapped_func = self._create_wrapper(original_func, torch.float32)
            
            # Set wrapped function as attribute of the module with the same name as original function
            setattr(module, func_name, wrapped_func)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restores original function
        """
        if not self.enabled:
            return

        self._depth -= 1
        if self._depth > 0:
            return False

        # Restore original functions
        for module, func_name in self.DOWNCAST_OPS + self.UPCAST_OPS:
            setattr(module, func_name, self._original_funcs[(module, func_name)])
        
        
        # Clear the stored functions for the next use
        self._original_funcs = {}
        return False


class StaticGradScaler:
    def __init__(self, scale):
        """
        scale: loss scaling coef
        """
        self._scale = scale
        self.inv_scale = 1.0 / scale
        
    def scale(self, loss):
        """Scales the loss"""
        scaled_loss = loss * self._scale
        return scaled_loss
    
    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        with torch.no_grad():
            grads = []
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue 
                    grads.append(param.grad)
            all_finite = True
            if grads:
                all_finite = torch.stack([torch.isfinite(grad).all() for grad in grads]).all().item()
            if not all_finite:
                optimizer.zero_grad()
                return
            for grad in grads:
                grad.mul_(self.inv_scale)
        optimizer.step()
        
    
    def update(self):
        """Updates scaling coef"""
        pass


class DynamicGradScaler:
    def __init__(self, scale, factor, patience, min_scale, max_scale):
        """
        scale: initial value of loss scaling coef
        factor: multiplier that used for scaling coef increase/decrease
        patience: how many iters there should be no nan/inf to increase scale
        min_scale: minimal allowed scaling coef value
        max_scale: maximal allowed scaling coef value
        """
        self._scale = scale
        self.factor = factor
        self.counter = 0
        self.patience = patience
        self.min_scale = min_scale
        self.max_scale = max_scale

    def scale(self, loss):
        """Scales the loss"""
        return loss * self._scale

    def step(self, optimizer):
        """
        Performs single optimization step
        """
        # Ignore parameters whose grad is None. Check every gradient before
        # unscaling any of them, then unscale without using .data.
        # Unscale the gradients
        # If there is any nan/inf in the gradient decrease scaling coef using factor
        # Note that scaling coef should be greater than min_scale
        # Perform optimizer step
        # Skip optimizer step if there is any nan/inf in the gradient
        # Do not forget that torch accumulates gradients
        with torch.no_grad():
            grads = []
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grads.append(param.grad)
            all_finite = True
            if grads:
                all_finite = torch.stack([torch.isfinite(grad).all() for grad in grads]).all().item()
            if not all_finite:
                optimizer.zero_grad()
                self._scale = max(self._scale / self.factor, self.min_scale)
                self.counter = 0
                return
            for grad in grads:
                grad.mul_(1 / self._scale)
        optimizer.step()
        self.counter += 1


    def update(self):
        """Updates scaling coef"""
        # If there was no any nan/inf in the gradient patience steps, increase scaling coef using factor
        # Note that scaling coef should be smaller than max_scale
        if self.counter == self.patience:
            self._scale = min(self._scale * self.factor, self.max_scale)
            self.counter = 0
