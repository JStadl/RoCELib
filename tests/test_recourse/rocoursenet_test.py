from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.evaluations.RobustnessProportionEvaluator import RobustnessProportionEvaluator
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.recourse_methods.RoCourseNet import RoCourseNet
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.evaluations.ValidityEvaluator import ValidityEvaluator



def test_rocoursenet() -> None:
    # Step 1: Initialize the model and dataset
    model = TrainablePyTorchModel(34, [8], 1)  # Neural network with input size 34, hidden layer [8], and output size 1
    dl = get_example_dataset("ionosphere")  # Load the "ionosphere" dataset


    dl.default_preprocess()  # Preprocess the dataset (e.g., scaling, normalization)

    trained_model = model.train(dl.X, dl.y)
    ct = ClassificationTask(trained_model, dl)


    recourse = RoCourseNet(ct)

    res = recourse.generate_for_all()

    val = ValidityEvaluator(ct)

    x = val.evaluate(res)

    assert x > 0.05
