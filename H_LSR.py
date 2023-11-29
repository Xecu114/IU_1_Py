import numpy as np
from scipy.optimize import least_squares


# Definiere eine Funktion, die die Summe der quadrierten Abweichungen berechnet
def residuals(p, y, x):
    return y - np.polyval(p, x)


# Definiere eine Funktion, die die Least-Square Regression durchführt
def least_square_regression(ideal_functions, noisy_functions):
    # Initialisiere das Ergebnis-Array
    result = np.zeros((50, 2))
    noise_free_functions = np.zeros(4)
    for j, f in enumerate(noisy_functions):
        # Iterate over the 50 ideal functions
        for i in range(50):
            # Perform the least square regression
            #
            # residuals is a function that calculates the difference between
            # the observed and predicted values of a dependent variable.
            # In this case, it is used to calculate the residuals of the
            # ideal_functions[i] with respect to the np.arange(400).
            #
            # np.ones(3) creates an array of shape (3,) with all elements set
            # to 1. It is used as an initial guess for the optimization
            # problem.
            #
            # The method parameter can also be set to 'lm' to use
            # the Levenberg-Marquardt algorithm
            #
            # The verbose parameter is set to 0 to suppress the output of
            # the solver.
            #
            # The x attribute of the returned object contains
            # the solution to the optimization problem.
            p = least_squares(residuals, np.ones(3), method='trf',
                              args=(ideal_functions[i], np.arange(400)),
                              verbose=0).x
            # print(i, " result: ", p)
            # Speichere das Ergebnis in das Ergebnis-Array
            result[i, 0] = i+1
            result[i, 1] = np.sum((np.polyval(p, np.arange(400)) - f) ** 2)

        # Sort the result array after the sum of squared deviations
        result = result[result[:, 1].argsort()]
        print("result Array: ", result)
        print(f"Funktion mit kleinsten Abweichungen: Y{int(result[0, 0])}")
        noise_free_functions[j] = int(result[0, 0])
    print(noise_free_functions)
    print(type(noise_free_functions[0]))


# Erstelle ein Array mit 50 inkludierten Arrays und jeweils 400 Datenpunkten
ideal_functions = np.random.rand(50, 400)
f = np.random.rand(4, 400)
print(f)
# Führe die Least-Square Regression durch
# least_square_regression(ideal_functions, f)
