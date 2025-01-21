import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.tasks.Task import Task

class RBR(RecourseGenerator):
    """
    Implements the Robust Bayesian Recourse (RBR) method for generating counterfactual explanations.
    """

    def __init__(self, ct: Task, num_components = 3, perturb_radius = 0.1, max_iter = 1000, num_cfacts = 10, random_state = None):
        """
            Initializes the RBRRecourseGenerator with a given task.

            @param ct: The task to solve, provided as a Task instance.
            @param num_components: Number of Gaussian components for probabilistic modeling.
            @param perturb_radius: Radius used to generate feasible counterfactual samples.
            @param max_iter: Maximum number of optimization iterations.
            @param num_cfacts: Number of counterfactual samples to generate.
            @param random_state: Random seed for reproducibility.
        """
        super().__init__(ct)
        self.num_components = num_components
        self.perturb_radius = perturb_radius
        self.max_iter = max_iter
        self.num_cfacts = num_cfacts
        self.random_state = random_state

        # Fit a Gaussian Mixture Model (GMM) to approximate data distribution
        self.gmm = GaussianMixture(n_components=self.num_components, covariance_type='full', random_state=random_state)
        self.gmm.fit(self.task.training_data.data.drop(columns=["target"], errors='ignore'))

    def generation_method(self, instance, column_name = "target", neg_value = 0, **kwargs) -> pd.DataFrame:
        """
            Generates a robust counterfactual explanation using probabilistic modeling.

            @param instance: The instance for which to generate a counterfactual. Can be a DataFrame or Series.
            @param column_name: The name of the target column.
            @param neg_value: The value considered negative in the target variable.
            @return: A DataFrame containing the robust counterfactual explanation.
        """

        instance_values = instance.drop(column = [column_name], errors = 'ignore').values.flatten()

        def objective_function(x):
            return np.linalg.norm(x - instance_values)

        # Define robustness constraint using Gaussian Mixture Model (GMM)
        def robustness_constraint(x):
            # Compute probability density
            prob_density = np.exp(self.gmm.score_samples([x]))
            # Ensure counterfactual is in a dense region
            return prob_density - 0.01

        # Initial counterfactual guess
        x0 = instance_values

        # Define optimization constraints
        constraints = ({'type': 'ineq', 'fun': robustness_constraint})

        # Solve min-max optimization problem using gradient-based optimization
        result = minimize(objective_function, x0, method="SLSQP", constraints=constraints,
                          options={'maxiter': self.max_iter})

        # If optimization is successful, return counterfactual; otherwise, return original instance
        if result.success:
            ce = pd.DataFrame([result.x], columns=instance.drop(columns=[column_name], errors='ignore').columns)
            return ce
        else:
            print("Failed to generate a feasible counterfactual.")
            return pd.DataFrame(instance).T
