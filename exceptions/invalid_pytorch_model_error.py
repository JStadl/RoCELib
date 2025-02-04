class InvalidPytorchModelError(TypeError):
    """Raised when a loaded model is not a valid PyTorch model."""

    def __init__(self, model = None):
        model_type = type(model).__name__ if model else "Unknown"
        message = (
            f"Expected a PyTorch model (torch.nn.Module), but got {model_type}"
            "The loaded model is not a torch model, but instead relies on a self defined class. \n"
            "Please save your model again, ensuring to save the underlying torch model, rather than your wrapper class\n"
            "Then try and load your model in again"
        )
        super().__init__(message)
