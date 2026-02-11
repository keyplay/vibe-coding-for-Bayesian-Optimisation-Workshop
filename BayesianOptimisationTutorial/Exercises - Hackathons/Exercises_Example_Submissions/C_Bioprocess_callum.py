import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel

# # Assume vl.conduct_experiment is available (provided by the hackathon environment)
# # For local testing, you might mock this function:
# class MockVl:
#     def conduct_experiment(self, X_input):
#         results = []
#         for row in X_input:
#             temp, ph, feed1, feed2, feed3, fidelity = row
#             # Simple mock function - replace with actual vl.conduct_experiment
#             # This mock is just to allow the code to run and demonstrate the structure
#             # Higher fidelity generally gives better (or at least more consistent) results
#             # and has a higher cost.
            
#             # Simple, somewhat arbitrary relationship for mocking purposes
#             base_titre = (temp - 35)**2 + (ph - 7)**2 + (feed1 + feed2 + feed3) / 5
            
#             if fidelity == 0: # Lowest fidelity
#                 cost = 10
#                 titre = 50 - base_titre + np.random.normal(0, 5) # More noisy
#                 # For fidelity 0, only feed2 is relevant
#                 titre = 50 - (temp - 35)**2 - (ph - 7)**2 - feed2/2 + np.random.normal(0, 5)

#             elif fidelity == 1: # Middle fidelity
#                 cost = 575
#                 titre = 80 - base_titre + np.random.normal(0, 2) # Less noisy
#                 titre = 80 - (temp - 35)**2 - (ph - 7)**2 - (feed1 + feed2 + feed3)/3 + np.random.normal(0, 2)

#             else: # Highest fidelity
#                 cost = 2100
#                 titre = 100 - base_titre + np.random.normal(0, 0.5) # Least noisy
#                 titre = 100 - (temp - 35)**2 - (ph - 7)**2 - (feed1 + feed2 + feed3)/5 + np.random.normal(0, 0.5)


#             results.append(max(0, titre)) # Titre cannot be negative for mock
        
#         # In a real scenario, this would likely be a single call that returns multiple results or a single result
#         # depending on X_input shape. For now, assume it takes an (N,6) array and returns (N,) array.
        
#         # This mock needs to return costs as well for budget management.
#         # Let's assume vl.conduct_experiment returns (titre, cost_incurred)
#         # For simplicity in this mock, I'll return just the titre for now,
#         # but the BO class needs to manage the cost separately based on the fidelity input.
#         return np.array(results)

# vl = MockVl() # Instantiate the mock for local testing

# # Define input bounds
# BOUNDS = {
#     'temperature': (30, 40),
#     'pH': (6, 8),
#     'feed1': (0, 50),
#     'feed2': (0, 50),
#     'feed3': (0, 50),
#     'fidelity': (0, 2) # Discrete values 0, 1, 2
# }

# FIDELITY_COSTS = {
#     0: 10,
#     1: 575,
#     2: 2100
# }

class GP:
    def __init__(self, X_train, Y_train):

        
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0]*6, nu=2.5) + WhiteKernel(noise_level=0.1)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)
        self.fit(X_train, Y_train)

    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X_test):
        mu, sigma = self.model.predict(X_test, return_std=True)
        return mu, sigma

class BO:
    def __init__(self, budget=10000, initial_points_count=6, obj_func=None, bounds=None, fidelity_costs=None):
        self.budget = budget
        self.initial_points_count = initial_points_count
        self.obj_func = obj_func
        self.bounds = bounds
        self.fidelity_costs = fidelity_costs

        self.X = [] # List to store all evaluated input points
        self.Y = [] # List to store all evaluated output (titre) values
        self.current_cost = 0

        self._initialize_data()

    def _initialize_data(self):

        initial_X_candidates = []
        
     
        initial_X = np.array([
            [35, 7.0, 25, 25, 25, 0], # Low fidelity
            [32, 6.5, 10, 15, 20, 0], # Low fidelity
            [38, 7.5, 40, 30, 10, 1], # Middle fidelity
            [33, 6.8, 5, 45, 35, 1], # Middle fidelity
            [36, 7.2, 20, 20, 20, 2], # High fidelity
            [34, 7.1, 15, 35, 25, 2]  # High fidelity
        ])
        
        # Check if the generated initial_X fits the initial_points_count, if not, adjust or sample more.
        if initial_X.shape[0] > self.initial_points_count:
            initial_X = initial_X[:self.initial_points_count]
        elif initial_X.shape[0] < self.initial_points_count:
            # Need to add more points. For simplicity, let's add random ones.
            num_to_add = self.initial_points_count - initial_X.shape[0]
            random_points = np.zeros((num_to_add, 6))
            for i in range(num_to_add):
                random_points[i, 0] = np.random.uniform(*self.bounds['temperature'])
                random_points[i, 1] = np.random.uniform(*self.bounds['pH'])
                random_points[i, 2] = np.random.uniform(*self.bounds['feed1'])
                random_points[i, 3] = np.random.uniform(*self.bounds['feed2'])
                random_points[i, 4] = np.random.uniform(*self.bounds['feed3'])
                random_points[i, 5] = np.random.choice([0, 1, 2]) # Random fidelity
            initial_X = np.vstack([initial_X, random_points])

        initial_Y = self.obj_func(initial_X) # Evaluate initial points

        self.X = initial_X.tolist() # Convert to list for appending
        self.Y = initial_Y.tolist()

        # Update current_cost based on initial evaluations
        for point in initial_X:
            self.current_cost += self.fidelity_costs[int(point[-1])] # Fidelity is the last element

        print(f"Initial cost incurred: {self.current_cost}")

    def _expected_improvement(self, X_candidate, gp_model, Y_max):
        # Calculate Expected Improvement (EI)
        mu, sigma = gp_model.predict(X_candidate.reshape(1, -1))
        
        # Handle cases where sigma is very small to avoid division by zero or large numbers
        if sigma <= 1e-6:
            return 0.0

        Z = (mu - Y_max) / sigma
        ei = (mu - Y_max) * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei[0]

    def _acquisition_function(self, X_candidate, gp_model, Y_max, fidelity_costs):
   
        

        X_candidate[-1] = int(np.round(np.clip(X_candidate[-1], 0, 2))) 
        
        cost = fidelity_costs[int(X_candidate[-1])]
        
 
        if cost == 0:
            return -self._expected_improvement(X_candidate, gp_model, Y_max) # Just maximize EI if cost is free

        ei = self._expected_improvement(X_candidate, gp_model, Y_max)
        
      
        return -ei / cost

    def _optimize_acquisition_function(self, gp_model, Y_max):
        # Optimize the acquisition function to find the next best point
        
        # Define bounds for the optimizer
        # Remember the order: [temperature, pH, feed1, feed2, feed3, fidelity]
        bds = np.array([
            self.bounds['temperature'],
            self.bounds['pH'],
            self.bounds['feed1'],
            self.bounds['feed2'],
            self.bounds['feed3'],
            self.bounds['fidelity']
        ])
        
        # Multi-start optimization to avoid local optima
        num_restarts = 10
        best_acquisition_value = -np.inf
        best_x_next = None

        for _ in range(num_restarts):
            # Generate random starting points within the bounds
            x0 = np.array([
                np.random.uniform(*self.bounds['temperature']),
                np.random.uniform(*self.bounds['pH']),
                np.random.uniform(*self.bounds['feed1']),
                np.random.uniform(*self.bounds['feed2']),
                np.random.uniform(*self.bounds['feed3']),
                np.random.uniform(*self.bounds['fidelity']) # Treat fidelity as continuous for optimization, then round
            ])

            result = minimize(
                fun=self._acquisition_function,
                x0=x0,
                args=(gp_model, Y_max, self.fidelity_costs),
                bounds=bds,
                method='L-BFGS-B' # Suitable for bounded, continuous problems
            )

            if -result.fun > best_acquisition_value: # Remember we are minimizing -EI/cost
                best_acquisition_value = -result.fun
                best_x_next = result.x

        # Round the fidelity to the nearest integer (0, 1, or 2)
        if best_x_next is not None:
            best_x_next[-1] = int(np.round(np.clip(best_x_next[-1], 0, 2)))

        return best_x_next

    def run_optimization(self):
        max_iterations = 50 # Cap iterations to prevent infinite loops if budget logic is off
        
        for iteration in range(max_iterations):
            if self.current_cost >= self.budget:
                print(f"Budget exhausted at iteration {iteration}. Total cost: {self.current_cost}")
                break

            # Find the current maximum observed titre
            Y_max = np.max(self.Y)

            # Fit GP model to all observed data
            gp_model = GP(np.array(self.X), np.array(self.Y))

            # Optimize acquisition function to find the next point and fidelity
            next_x = self._optimize_acquisition_function(gp_model, Y_max)
            
            if next_x is None:
                print("Could not find a valid next point. Stopping optimization.")
                break

            fidelity_to_evaluate = int(next_x[-1])
            cost_of_next_evaluation = self.fidelity_costs[fidelity_to_evaluate]

            # Check if budget allows this evaluation
            if self.current_cost + cost_of_next_evaluation > self.budget:
                print(f"Cannot afford fidelity {fidelity_to_evaluate} (cost {cost_of_next_evaluation}). Trying lower fidelities.")
                # Try to find the highest affordable fidelity
                affordable_fidelity = -1
                for f_level in sorted(self.fidelity_costs.keys(), reverse=True):
                    if self.current_cost + self.fidelity_costs[f_level] <= self.budget:
                        affordable_fidelity = f_level
                        break
                
                if affordable_fidelity != -1:
                    print(f"Switching to affordable fidelity: {affordable_fidelity}")
                    next_x[-1] = affordable_fidelity
                    cost_of_next_evaluation = self.fidelity_costs[affordable_fidelity]
                else:
                    print("No affordable fidelity found. Stopping optimization.")
                    break

            # Evaluate the chosen point
            new_Y = self.obj_func(next_x.reshape(1, -1))[0] # Assuming obj_func takes (N,6) and returns (N,)

            # Update data and cost
            self.X.append(next_x.tolist())
            self.Y.append(new_Y)
            self.current_cost += cost_of_next_evaluation

            print(f"Iteration {iteration + 1}: Evaluated {next_x.tolist()}, Titre: {new_Y:.2f}, Cost: {cost_of_next_evaluation}, Total Cost: {self.current_cost}")

        print("\nOptimization complete.")
        best_titre_idx = np.argmax(self.Y)
        print(f"Best Titre Found: {self.Y[best_titre_idx]:.2f} at parameters: {self.X[best_titre_idx]}")


# Define the objective function wrapper
def obj_func(X):
 
    return np.array(vl.conduct_experiment(X))


search_space_bounds = np.array([
    BOUNDS['temperature'],
    BOUNDS['pH'],
    BOUNDS['feed1'],
    BOUNDS['feed2'],
    BOUNDS['feed3'],

    BOUNDS['fidelity']
])


BO_m = BO(
    budget=10000,
    initial_points_count=6,
    obj_func=obj_func,
    bounds=BOUNDS,
    fidelity_costs=FIDELITY_COSTS
)

BO_m.run_optimization()