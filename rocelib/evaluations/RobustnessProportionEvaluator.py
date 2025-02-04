import pandas as pd
from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator
from rocelib.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.robustness_evaluations.ModelChangesRobustnessEvaluator import ModelChangesRobustnessEvaluator


class RobustnessProportionEvaluator(RecourseEvaluator):
    """
     An Evaluator class which evaluates the proportion of recourses which are robust

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the robustness of CEs

    robustness_evaluator: ModelChangesRobustnessEvaluator
        An instance of ModelChangesRobustnessEvaluator to evaluate the robustness of the CEs

    valid_val: int
        Stores what the target value of a valid counterfactual is defined as

    target_col: str
        Stores what the target column name is

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are robust for the given parameters

    -------
    """

    def evaluate(self, recourses, delta=0.05, bias_delta=0, M=1000000, epsilon=0.001, valid_val=1, column_name="target",
                 robustness_evaluator: ModelChangesRobustnessEvaluator.__class__ = DeltaRobustnessEvaluator, **kwargs):
        """Evaluates the proportion of CEs which are robust for the given parameters."""

        if recourses is None or not isinstance(recourses, pd.DataFrame) or recourses.empty:
            raise ValueError("Expected 'recourses' to be a non-empty pandas DataFrame.")

        instances = recourses.drop(columns=[column_name, "loss"], errors='ignore')

        robustness_evaluator = robustness_evaluator(self.task)

        robust = sum(
            robustness_evaluator.evaluate(instance, desired_output=valid_val, delta=delta, bias_delta=bias_delta, M=M,
                                          epsilon=epsilon)
            for _, instance in instances.iterrows() if instance is not None
        )

        return robust / len(instances)
