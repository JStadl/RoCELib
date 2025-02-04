import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from rocelib.evaluations.RecourseEvaluator import RecourseEvaluator


### Work In Progress ###
class ManifoldEvaluator(RecourseEvaluator):
    """
     An Evaluator class which evaluates the proportion of recourses which are on the data manifold using LOF

        ...

    Attributes / Properties
    -------

    task: Task
        Stores the Task for which we are evaluating the robustness of CEs

    -------

    Methods
    -------

    evaluate() -> int:
        Returns the proportion of CEs which are robust for the given parameters

    -------
    """

    def evaluate(self, recourses, n_neighbors=20, column_name="target", **kwargs):
        """Determines the proportion of CEs that lie on the data manifold based on LOF."""

        if recourses is None or recourses.empty:
            raise ValueError("Recourse dataset is empty. Cannot perform manifold evaluation.")

        data = self.task.training_data.X
        recourses = recourses.drop(columns=[column_name, "loss"], errors='ignore')

        on_manifold = 0
        cnt = 0

        for _, recourse in recourses.iterrows():
            if recourse is None or recourse.isnull().any():
                continue  # Skip invalid rows

            # Combine the dataset with the new instance
            data_with_instance = pd.concat([data, pd.DataFrame([recourse])], ignore_index=True)
            data_with_instance.columns = data_with_instance.columns.astype(str)

            try:
                lof = LocalOutlierFactor(n_neighbors=n_neighbors)
                lof_scores = lof.fit_predict(data_with_instance)
                instance_lof_score = lof.negative_outlier_factor_[-1]
            except Exception as e:
                raise RuntimeError(f"LOF calculation failed: {e}")

            if instance_lof_score > 0.9:
                on_manifold += 1
            cnt += 1

        return on_manifold / cnt if cnt > 0 else 0
