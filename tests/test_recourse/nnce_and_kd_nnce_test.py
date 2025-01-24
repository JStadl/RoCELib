from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
from rocelib.models.Models import get_sklearn_model
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.KDTreeNNCE import KDTreeNNCE
from rocelib.recourse_methods.NNCE import NNCE
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.tasks.ClassificationTask import ClassificationTask
from ..test_constants import TEST_INSTANCES, PASS_THRESHOLD


def test_kdtree_nnce() -> None:
    model = get_sklearn_model("log_reg")
    dl = get_example_dataset("ionosphere")
    dl.default_preprocess()

    ct = ClassificationTask(model, dl)

    ct.train()

    recourse = KDTreeNNCE(ct)

    res = recourse.generate_for_all(neg_value=0)
    val = ValidityEvaluator(ct)
    validity_pct = val.evaluate(res, test_instances=TEST_INSTANCES)

    assert validity_pct > PASS_THRESHOLD


def test_kdtree_nnce_same_as_nnce() -> None:
    model = SimpleNNModel(10, [7], 1)
    dl = CsvDatasetLoader('./assets/recruitment_data.csv', "HiringDecision")

    ct = ClassificationTask(model, dl)

    ct.train()

    kdrecourse = KDTreeNNCE(ct)

    nncerecourse = NNCE(ct)
    negs = dl.get_negative_instances(neg_value=0, column_name="HiringDecision")

    for _, neg in negs.iterrows():
        a = kdrecourse.generate_for_instance(neg)
        b = nncerecourse.generate_for_instance(neg)

        assert a.index == b.index
