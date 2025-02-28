from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.tasks.TaskBuilder import TaskBuilder


def test_mce_predicts_positive_instances(testing_models):  # TODO
    tb = TaskBuilder()
    dl_training = get_example_dataset("ionosphere")
    dl = dl_training
    tb.add_pytorch_model(34, [8], 1, dl_training)
    tb.add_pytorch_model(34, [16, 8], 1, dl_training)
    tb.add_pytorch_model(34, [16, 8, 4], 1, dl_training)

    tb.add_data(dl)
    ct = tb.build()

    res = ct.generate_mm(["MMMILP"])

    assert not res["MMMILP"][0].empty
