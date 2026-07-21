class MedBriefError(Exception):
    status_code = 400
    code = "medbrief_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidFileError(MedBriefError):
    code = "invalid_file"


class NotFoundError(MedBriefError):
    status_code = 404
    code = "not_found"


class ProcessingError(MedBriefError):
    status_code = 422
    code = "processing_error"


class ModelUnavailableError(MedBriefError):
    status_code = 503
    code = "model_unavailable"

