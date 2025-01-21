from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.Wachter import Wachter
from rocelib.tasks.ClassificationTask import ClassificationTask
from ..test_constants import TEST_INSTANCES, PASS_THRESHOLD


def test_wachter() -> None:

    dl = get_example_dataset("ionosphere")

    model = SimpleNNModel(34, [8], 1)

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = Wachter(ct)

    res = recourse.generate_for_all(neg_value=0)
    val = ValidityEvaluator(ct)
    validty_pct = val.evaluate(res, test_instances=TEST_INSTANCES)

    assert validty_pct >= PASS_THRESHOLD


