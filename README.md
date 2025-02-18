# Haper-Agent

This is the AI Agent of haper that responsible for processing message and generate report using LLMs.

---


## Requirements

To get started, ensure you have the following installed:

- Python 3.7 (or a compatible version).

### DB and other storage resource requirement

- Postgresql
- AWS SQS

## Usage

1. This project use python-dotenv to load runtime envs. First create your own .env file to the root of this project
   ```text
   LOG_LEVEL=<log level, DEBUG,INFO, WARNING, etc>
   SQS_REGION=<your sqs resource region>
   SQS_ENDPOINT=<your sqs endpoint>
   SQS_QUEUE_URL=<your sqs queue url>
   AWS_ACCESS_KEY_ID=<your sqs access key id>
   AWS_SECRET_ACCESS_KEY=<your sqs secret access key>
   ```

2. Create Virtual Env or using exist one and install pip dependencies
   ```bash
   python3 -m venv ./venv
   source ./venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   python app.py
   ```

---

## License

This project is licensed under the [AGPL License](LICENSE). Feel free to use and modify it as needed.
