import pandas as pd

from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.STCE import TrexNN
from rocelib.tasks.ClassificationTask import ClassificationTask
from ..test_constants import TEST_INSTANCES, PASS_THRESHOLD


def test_stce() -> None:
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = TrexNN(ct)
    re = RobustnessProportionEvaluator(ct)
    ces = []

    counterfactuals = 0

    for _, neg in dl.get_negative_instances(neg_value=0).head(TEST_INSTANCES).iterrows():
        res = recourse.generate_for_instance(neg, delta=0.005)
        ces.append(res)
        if model.predict_single(res):
            counterfactuals += 1

    assert counterfactuals >= TEST_INSTANCES * PASS_THRESHOLD
    ce_df = pd.concat(ces)
    print(re.evaluate(ce_df, delta=0.005))
