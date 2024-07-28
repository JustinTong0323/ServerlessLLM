# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import os
from argparse import Namespace, _SubParsersAction

import requests

from serverless_llm.cli._cli_utils import read_config, validate_config
from serverless_llm.serve.logger import init_logger

logger = init_logger(__name__)


class DeployCommand:
    @staticmethod
    def register_subcommand(parser: _SubParsersAction):
        deploy_parser = parser.add_parser(
            "deploy", help="Deploy a model using a config file or model name."
        )
        deploy_parser.add_argument(
            "--model",
            type=str,
            help="Model name to deploy with default configuration.",
        )
        deploy_parser.add_argument(
            "--config", type=str, help="Path to the JSON config file."
        )
        deploy_parser.add_argument(
            "--backend", type=str, help="Overwrite the backend in the default configuration."
        )
        deploy_parser.add_argument(
            "--num_gpus", type=int, help="Overwrite the number of GPUs in the default configuration."
        )
        deploy_parser.add_argument(
            "--target", type=int, help="Overwrite the target concurrency in the default configuration."
        )
        deploy_parser.add_argument(
            "--min_instances", type=int, help="Overwrite the minimum instances in the default configuration."
        )
        deploy_parser.add_argument(
            "--max_instances", type=int, help="Overwrite the maximum instances in the default configuration."
        )
        deploy_parser.set_defaults(func=DeployCommand)

    def __init__(self, args: Namespace) -> None:
        self.model = args.model
        self.config_path = args.config
        self.backend = args.backend
        self.num_gpus = args.num_gpus
        self.target = args.target
        self.min_instances = args.min_instances
        self.max_instances = args.max_instances
        self.url = (
            os.getenv("LLM_SERVER_URL", "http://localhost:8343/") + "register"
        )
        self.default_config_path = os.path.join(
            os.path.dirname(__file__), "default_config.json"
        )

        self.validate_args()

    def validate_args(self) -> None:
        """Validate the provided arguments to ensure correctness."""
        if self.num_gpus is not None and self.num_gpus < 0:
            raise ValueError("Number of GPUs cannot be negative.")
        if self.target is not None and self.target < 0:
            raise ValueError("Target concurrency cannot be negative.")
        if self.min_instances is not None and self.min_instances < 0:
            raise ValueError("Minimum instances cannot be negative.")
        if self.max_instances is not None and self.max_instances < 0:
            raise ValueError("Maximum instances cannot be negative.")
        if self.min_instances is not None and self.max_instances is not None:
            if self.min_instances > self.max_instances:
                raise ValueError("Minimum instances cannot be greater than maximum instances.")

    def run(self) -> None:
        if self.config_path:
            config_data = read_config(self.config_path)
            validate_config(config_data)
            self.deploy_model(config_data)
        elif self.model:
            config_data = read_config(self.default_config_path)
            config_data["model"] = self.model
            config_data["backend_config"]["pretrained_model_name_or_path"] = (
                self.model
            )
            if self.backend:
                config_data["backend"] = self.backend
            if self.num_gpus is not None:
                config_data["num_gpus"] = self.num_gpus
            if self.target is not None:
                config_data["auto_scaling_config"]["target"] = self.target
            if self.min_instances is not None:
                config_data["auto_scaling_config"]["min_instances"] = self.min_instances
            if self.max_instances is not None:
                config_data["auto_scaling_config"]["max_instances"] = self.max_instances

            logger.info(
                f"Deploying model {self.model} with custom configuration."
            )
            self.deploy_model(config_data)
        else:
            logger.error("You must specify either --model or --config.")
            exit(1)

    def deploy_model(self, config_data: dict) -> None:
        headers = {"Content-Type": "application/json"}

        # Send POST request to the /register endpoint
        response = requests.post(self.url, headers=headers, json=config_data)

        if response.status_code == 200:
            logger.info("Model registered successfully.")
        else:
            logger.error(
                f"Failed to register model. Status code: {response.status_code}. Response: {response.text}"
            )
