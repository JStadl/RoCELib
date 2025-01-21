from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.RNCE import RNCE
from rocelib.tasks.ClassificationTask import ClassificationTask
from ..test_constants import TEST_INSTANCES, PASS_THRESHOLD


def test_rnce() -> None:
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    delta = 0.01

    recourse = RNCE(ct)
    counterfactuals = 0

    for _, neg in dl.get_negative_instances(neg_value=0).head(TEST_INSTANCES).iterrows():
        res = recourse.generate_for_instance(neg, delta=delta)
        if recourse.intabs.evaluate(res, delta=delta):
            counterfactuals += 1

    assert counterfactuals >= TEST_INSTANCES * PASS_THRESHOLD
