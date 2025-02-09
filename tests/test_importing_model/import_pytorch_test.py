from exceptions.invalid_pytorch_model_error import InvalidPytorchModelError
from rocelib.datasets.ExampleDatasets import get_example_dataset
from rocelib.models.pytorch_models.TrainablePyTorchModel import TrainablePyTorchModel
from rocelib.tasks.ClassificationTask import ClassificationTask
from rocelib.tasks.TaskBuilder import TaskBuilder

from rocelib.models.imported_models.PytorchModel import PytorchModel
import torch
import os
import pytest


def test_imported_pytorch_model_file_predict_single_same_as_original(testing_models) -> None:
    # Create Model
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    # Save Model
    torch.save(ct.model.model, "./model.pt")

    # Import Model

    imported_model = PytorchModel("./model.pt")

    for _, instance in ct.dataset.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == imported_model.predict_single(instance_x)

    os.remove('./model.pt')


def test_imported_pytorch_model_file_predict_all_same_as_original(testing_models) -> None:
    # Create Model
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    # Save Model
    torch.save(ct.model.model, "./model.pt")

    # Import Model
    trained_model = PytorchModel("./model.pt")

    predictions_1 = ct.model.predict(ct.dataset.data.drop("target", axis=1))
    predictions_2 = trained_model.predict(ct.dataset.data.drop("target", axis=1))
    assert predictions_1.all() == predictions_2.all()

    os.remove('./model.pt')


def test_imported_pytorch_model_from_instance_predict_single_same_as_original(testing_models) -> None:
    # Create Model
    ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)

    torch_model = ct.model.model

    # Import Model
    trained_model = PytorchModel.from_model(torch_model)

    for _, instance in ct.dataset.data.iterrows():
        instance_x = instance.drop("target")
        assert ct.model.predict_single(instance_x) == trained_model.predict_single(instance_x)


def test_throws_file_not_found_error() -> None:
    with pytest.raises(FileNotFoundError):
        trained_model = PytorchModel("./garbage.pt")


def test_throws_wrong_file_type_error() -> None:
    with pytest.raises(TypeError):
        trained_model = PytorchModel("./test.txt")


def test_throws_error_if_underlying_model_not_pytorch(testing_models) -> None:
    with pytest.raises(InvalidPytorchModelError):
        ct = testing_models.get("ionosphere", "ionosphere", "pytorch", 34, 8, 1)
        # Save The SimpleNNModel rather than Torch Model
        torch.save(ct.model, "./model.pt")
        # Import Model
        trained_model = PytorchModel("./model.pt")

        os.remove('./model.pt')

