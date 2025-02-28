import numpy as np
import pandas as pd
import torch
from test_helpers.TestingModels import TestingModels  # Import your testing_models class
from enums.dataset_enums import Dataset
from enums.model_enums import ModelType
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.RoCourseNet import RoCourseNet
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.tasks.TaskBuilder import TaskBuilder

def test_correct_recourses_generated_for(testing_models) -> None:
    # ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
    # ces = ct.generate([
    # # "BinaryLinearSearch",
    # # "GuidedBinaryLinearSearch",
    # # "NNCE",
    # # "KDTreeNNCE",
    # # "MCE",
    # # "Wachter",
    # "RNCE",
    # "MCER",
    # # "STCE"
    # ])
    # evals = ct.evaluate([
    # # "BinaryLinearSearch",
    # # "GuidedBinaryLinearSearch",
    # # "NNCE",
    # # "KDTreeNNCE",
    # # "MCE",
    # # "Wachter",
    # "RNCE",
    # "MCER",
    # # "STCE"
    # ], [
    #     "Distance",
    #         "Validity",
    #         "ManifoldEvaluator",
    #         "RobustnessProportionEvaluator",
    #         "ModelMultiplicityRobustness",
    #         "DeltaRobustnessEvaluator",
    #         "InvalidationRateRobustnessEvaluator"
    #     ])
    dl = get_example_dataset("ionosphere")
    ct = TaskBuilder().add_pytorch_model(34, [8], 1, dl).add_pytorch_model(34, [8], 1, dl).add_data(dl).build()
    ces = ct.generate_mm([
    # "BinaryLinearSearch",
    # "GuidedBinaryLinearSearch",
    # "NNCE",
    # "KDTreeNNCE",
    # "MCE",
    # "Wachter",
    "RNCE",
    "MCER",
    # "STCE"
    ])
    evals = ct.evaluate([
    # "BinaryLinearSearch",
    # "GuidedBinaryLinearSearch",
    # "NNCE",
    # "KDTreeNNCE",
    # "MCE",
    # "Wachter",
    "RNCE",
    "MCER",
    # "STCE"
    ], [
        "ModelMultiplicityRobustness"
        # "Distance",
        #     "Validity",
        #     "ManifoldEvaluator",
        #     "RobustnessProportionEvaluator",
        #     "ModelMultiplicityRobustness",
        #     "DeltaRobustnessEvaluator",
        #     "InvalidationRateRobustnessEvaluator"
        ])
    # recourse_methods = ["KDTreeNNCE"]
    # ces = ct.generate_mm(recourse_methods) # we are not generating for all methods we want to evaluate for

    # evals = ct.evaluate(["KDTreeNNCE"], ["ModelMultiplicityRobustness"])




if __name__ == "__main__":
    testing_models = TestingModels()  # Initialize your testing_models
    test_correct_recourses_generated_for(testing_models)