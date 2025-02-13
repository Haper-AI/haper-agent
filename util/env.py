import os
import dotenv

dotenv.load_dotenv()


class RuntimeEnv:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RuntimeEnv, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    @staticmethod
    def Instance():
        if not RuntimeEnv._instance:
            RuntimeEnv()
        return RuntimeEnv._instance

    def __init__(self):
        self.SQS_REGION = os.getenv("SQS_REGION")
        self.SQS_ENDPOINT = os.getenv("SQS_ENDPOINT")
        self.SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")