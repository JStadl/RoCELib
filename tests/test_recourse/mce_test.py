from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from rocelib.recourse_methods.MCE import MCE
from rocelib.tasks.ClassificationTask import ClassificationTask
from ..test_constants import TEST_INSTANCES, PASS_THRESHOLD


def test_mce_predicts_positive_instances():
    model = SimpleNNModel(34, [8], 1)
    dl = get_example_dataset("ionosphere")

    ct = ClassificationTask(model, dl)

    dl.default_preprocess()
    ct.train()

    recourse = MCE(ct)
    counterfactuals = 0

    for _, neg in dl.get_negative_instances(neg_value=0).head(TEST_INSTANCES).iterrows():

        res = recourse.generate_for_instance(neg)

        if not res.empty:
            prediction = model.predict_single(res)

            if prediction:
                counterfactuals += 1

    assert counterfactuals >= TEST_INSTANCES * PASS_THRESHOLD
