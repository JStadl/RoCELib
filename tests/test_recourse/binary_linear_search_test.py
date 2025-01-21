import numpy as np
import pandas as pd

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.BinaryLinearSearch import BinaryLinearSearch
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from ..test_constants import TEST_INSTANCES, PASS_THRESHOLD
from rocelib.tasks.ClassificationTask import ClassificationTask


def test_binary_linear_search_nn() -> None:
    # Create a new classification task and train the model on our data
    model = SimpleNNModel(10, [7], 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")
    ct = ClassificationTask(model, dl)

    ct.train()

    # Use BinaryLinearSearch to generate a recourse for each negative value
    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0)
    val = ValidityEvaluator(ct)
    validity_pct = val.evaluate(res, test_instances=TEST_INSTANCES)

    assert validity_pct > PASS_THRESHOLD


def test_binary_linear_search_dt() -> None:
    model = get_sklearn_model("decision_tree")
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = BinaryLinearSearch(ct)

    res = recourse.generate_for_all(neg_value=0)
    val = ValidityEvaluator(ct)
    validity_pct = val.evaluate(res, test_instances=TEST_INSTANCES)

    assert validity_pct > PASS_THRESHOLD


def test_binary_linear_search_lr() -> None:
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    def euclidean_copy(x: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
        return np.sqrt(np.sum((x.values - c.values) ** 2))

    recourse = BinaryLinearSearch(ct, custom_distance_func=euclidean_copy)

    res = recourse.generate_for_all(neg_value=0)
    val = ValidityEvaluator(ct)
    validity_pct = val.evaluate(res, test_instances=TEST_INSTANCES)

    assert validity_pct > PASS_THRESHOLD
