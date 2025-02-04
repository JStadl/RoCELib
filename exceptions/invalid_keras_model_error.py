class InvalidKerasModelError(TypeError):
    """Raised when a loaded model is not a valid Keras model."""

    def __init__(self, model=None):
        model_type = type(model).__name__ if model else "Unknown"
        message = (
            f"Expected a Keras model (keras.Model), but got {model_type}.\n"
            "Please make sure you are passing a valid trained Keras model."
        )
        super().__init__(message)