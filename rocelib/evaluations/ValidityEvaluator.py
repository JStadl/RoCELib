import pandas as pd

from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator


class ValidityEvaluator(RecourseEvaluator):
    """
     An Evaluator class which evaluates the proportion of recourses which are valid

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the validity of CEs

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are valid

    -------
    """

    def checkValidity(self, instance, valid_val):
        """Checks if a given CE is valid."""

        if instance is None:
            return False

        if not isinstance(instance, (pd.Series, pd.DataFrame)):
            raise TypeError(f"Expected 'instance' to be a pandas Series or DataFrame, but got {type(instance)}.")

        instance = pd.DataFrame(instance).T if not isinstance(instance, pd.DataFrame) else instance

        try:
            return self.task.model.predict_single(instance) == valid_val
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")


    def evaluate(self, recourses, valid_val=1, column_name="target", **kwargs):
        """
        Evaluates the proportion of CEs are valid
        @param recourses: pd.DataFrame, set of CEs which we want to evaluate
        @param valid_val: int, target column value which denotes a valid instance
        @param column_name: str, name of target column
        @param kwargs: other arguments
        @return: int, proportion of CEs which are valid
        """
        valid = 0
        cnt = 0

        # Remove redundant columns
        instances = recourses.drop(columns=[column_name, "loss"], errors='ignore')

        for _, instance in instances.iterrows():

            # Increment validity counter if CE is valid
            if instance is not None and not instance.empty and self.checkValidity(instance, valid_val):
                valid += 1

            # Increment total counter
            cnt += 1

        return valid / cnt
