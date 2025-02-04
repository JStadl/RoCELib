class InvalidSKLearnModelError(TypeError):
    """Raised when a loaded model is not a valid scikit-learn model."""

    def __init__(self, model=None):
        model_type = type(model).__name__ if model else "Unknown"
        message = (
            f"Expected an sklearn model (BaseEstimator), but got {model_type}.\n"
            "Please ensure you are passing a valid trained scikit-learn model."
        )
        super().__init__(message)