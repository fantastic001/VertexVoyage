
def grid_search(
        f, 
        apply, 
        acc, 
        param_ranges, 
        fixed_params=None, 
        intermediate_callback=None
    ):
    """
    Perform a grid search over specified parameter ranges and calculate accumulated results.
    
    Formally, result is as follows:

    $$ \text{result} = \bigoplus_{p_1 \in P_1} \bigoplus_{p_2 \in P_2} ... \bigoplus_{p_n \in P_n} \alpha(f(p_1, p_2, ..., p_n, \text{fixed\_params})) $$

    where \( P_i \) is the set of values for parameter \( p_i \), $\alpha$ is the apply function, and \( \bigoplus \) is the accumulation operation.

    Parameters:
    f : function
        The function to evaluate, which takes parameters and returns a result.
    apply : function
        A function that processes the output of `f`.
    acc : function
        A binary function that accumulates results.
    param_ranges : dict
        A dictionary where keys are parameter names and values are lists of parameter values.
    fixed_params : dict, optional
        A dictionary of parameters that remain constant during the search.
    intermediate_callback : function, optional
        A function that is called with the current result after each evaluation of `f` where current parameters are passed to callback as kwargs.

    Returns:
    The accumulated result after applying `f` over all combinations of parameters in `param_ranges`.
    """
    from itertools import product

    if fixed_params is None:
        fixed_params = {}

    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]

    result = None

    for values in product(*param_values):
        params = dict(zip(param_names, values))
        params.update(fixed_params)
        output = f(**params)
        if intermediate_callback is not None:
            intermediate_callback(output, **params)
        applied_output = apply(output)
        
        if result is None:
            result = applied_output
        else:
            result = acc(result, applied_output)

    return result