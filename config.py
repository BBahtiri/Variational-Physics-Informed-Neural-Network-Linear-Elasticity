# Configuration parameters
CONFIG = {
    # Network parameters
    'NUM_HIDDEN_LAYERS': 5,
    'NEURONS_PER_LAYER': 40,
    'LEARNING_RATE': 0.001,
    'LR_DECAY_RATE': 0.95,
    'LR_DECAY_STEPS': 500,
    
    # Material properties
    'YOUNGS_MODULUS': 1000.0,
    'POISSON_RATIO': 0.3,
    
    # Numerical parameters
    'NUM_QUAD_POINTS': 20,
    'NUM_BOUNDARY_POINTS': 200,
    'TEST_FUNCTION_START': 1,
    'TEST_FUNCTION_END': 25,
    
    # Training parameters
    'NUM_EPOCHS': 2000,
    'BOUNDARY_WEIGHT': 50.0,
    'GRADIENT_CLIP': 0.5,
    'USE_LBFGS': True,
    'LBFGS_MAX_ITER': 1000,
    
    # Prescribed displacement
    'DISPLACEMENT_X': 0.1,

    'U_BC_LEFT': 0.0,          # Displacement u at x = 0
    'V_BC_BOTTOM': 0.0,        # Displacement v at y = 0
    'DISPLACEMENT_X': 0.1,     # Prescribed displacement u at x = 1
    'ADF_ALPHA': 1.0,          # Exponent alpha in the ADF for x
    'ADF_BETA': 1.0,           # Exponent beta in the ADF for y
    'ADF_GAMMA': 1.0,          # Exponent gamma in the ADF for (1 - x)
    'GAMMA_DELTA': 1.0,        # Exponent delta in g_u(x)
    'ADF_BETA_V': 1.0,         # Exponent beta in the ADF for v
}