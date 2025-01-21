import pandas as pd
import torch

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.RBR import RBR
from rocelib.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
from rocelib.tasks.ClassificationTask import ClassificationTask

def test_rbr_generates_valid_counterfactuals():
    model = SimpleNNModel(input_dim = 34, hidden_dim = [10], output_dim = 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)
    dl.default_preprocess()
    ct.train()

    recourse = RBR(ct)

    for _, neg in dl.get_negative_instances(neg_value=0).iterrows():
        instance = pd.DataFrame([neg])
        res = recourse._generation_method(instance)

        if not res.empty:
            prediction = model.predict_single(res.to_numpy().reshape(1, -1))
            assert prediction
