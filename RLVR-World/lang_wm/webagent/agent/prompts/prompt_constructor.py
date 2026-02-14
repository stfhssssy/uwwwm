import json
import re
from pathlib import Path
from typing import Any, Generic, TypedDict, TypeVar, cast, List, Dict
from typing_extensions import Unpack
from dataclasses import dataclass
from PIL import Image

from browser_env import Action, ActionParsingError, Trajectory
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo, pil_to_b64, pil_to_vertex
from llms import lm_config
from llms.tokenizers import Tokenizer
from llms.utils import APIInput


class InstructionExample(TypedDict):
    ...

class ChatExample(InstructionExample):
    user: str
    assistant: str

class CompletionExample(InstructionExample):
    observation: str
    action: str

class MultimodalExample(InstructionExample):
    image_path: str

class MultimodalChatExample(ChatExample, MultimodalExample):
    ...

class MultimodalCompletionExample(CompletionExample, MultimodalExample):
    ...


_T = TypeVar("_T", bound=InstructionExample)
# class Instruction(TypedDict, Generic[_T]):
#     """Instruction for constructing prompt"""
#     intro: str
#     examples: list[_T]
#     template: str
#     meta_data: dict[str, Any]

T = TypeVar('T')
class InstructionDict(TypedDict):
    intro: str
    examples: List[Any]
    template: str
    meta_data: Dict[str, Any]

class Instruction(Generic[T]):
    def __init__(self, data: InstructionDict):
        self.intro: str = data['intro']
        self.examples: List[T] = data['examples']
        self.template: str = data['template']
        self.meta_data: Dict[str, Any] = data['meta_data']

class PromptConstructor(object):
    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        self.instruction_path = Path(instruction_path)
        self.obs_modality = "text"
        self.lm_config = lm_config

        self.instruction: Instruction = json.load(open(self.instruction_path))
        self.tokenizer = tokenizer

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[InstructionExample],
        current: str,
        *args,
        **kwargs
    ) -> APIInput:
        """Return the require format for an API"""
        message: list[dict[str, str]] | str
        if "openai" in self.lm_config.provider:
            example: InstructionExample
            match self.lm_config.mode:
                case "chat":
                    message = [{"role": "system", "content": intro}]
                    for example in cast(list[ChatExample], examples):
                        message += [
                            {
                                "role": "system",
                                "name": "example_user",
                                "content": example["user"],
                            },
                            {
                                "role": "system",
                                "name": "example_assistant",
                                "content": example["assistant"],
                            }
                        ]
                    message.append({"role": "user", "content": current})
                    return message

                case "completion":
                    message = f"{intro}\n\n"
                    message += "Here are a few examples:\n"
                    for example in cast(list[CompletionExample], examples):
                        message += f"Observation\n:{example['observation']}\n\n"
                        message += f"Action: {example['action']}\n\n"
                    message += "Now make prediction given the observation\n\n"
                    message += f"Observation\n:{current}\n\n"
                    message += "Action:"
                    return message

                case _:
                    raise ValueError(
                        f"OpenAI models do not support mode {self.lm_config.mode}"
                    )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
        *args,
        **kwargs
    ) -> APIInput:
        raise NotImplementedError

    def map_url_to_real(self, url: str) -> str:
        """Map the urls to their real world counterparts"""
        for i, j in URL_MAPPINGS.items():
            if i in url:
                url = url.replace(i, j)
        return url

    def map_url_to_local(self, url: str) -> str:
        """Map the urls to their local counterparts"""
        for i, j in URL_MAPPINGS.items():
            if j in url:
                url = url.replace(j, i)
            # https
            if j.replace("http", "https") in url:
                url = url.replace(j.replace("http", "https"), i)
        return url

    def _extract_action(self, response: str) -> str:
        raise NotImplementedError

    def extract_action(self, response: str) -> str:
        response = self._extract_action(response)
        response = self.map_url_to_local(response)
        return response


class DirectPromptConstructor(PromptConstructor):
    """The agent will direct predict the action"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {}
    ) -> APIInput:
        """Construct prompt given the trajectory"""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]

        # input x
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        # make sure all keywords are replaced
        assert all([f"{{k}}" not in current for k in keywords])
        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f"Cannot parse action from response {response}"
            )


class CoTPromptConstructor(PromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
        *args,
        **kwargs
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(intro, examples, current)
        return prompt

    def _extract_action(self, response: str) -> str:
        # find the first occurence of action
        action_splitter = self.instruction["meta_data"]["action_splitter"]
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ActionParsingError(
                f'Cannot find the answer phrase "{self.answer_phrase}" in "{response}"'
            )


class _MultimodalCoTPromptConstructorKwargs(TypedDict):
    page_screenshot_img: Image.Image
    images: list[Image.Image]


class MultimodalCoTPromptConstructor(CoTPromptConstructor):
    """The agent will perform step-by-step reasoning before the answer"""

    def __init__(
        self,
        instruction_path: str | Path,
        lm_config: lm_config.LMConfig,
        tokenizer: Tokenizer,
    ):
        super().__init__(instruction_path, lm_config, tokenizer)
        self.answer_phrase = self.instruction["meta_data"]["answer_phrase"]

    def construct(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any] = {},
        **kwargs: Unpack[_MultimodalCoTPromptConstructorKwargs]
    ) -> APIInput:
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]
        keywords = self.instruction["meta_data"]["keywords"]
        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]

        page_screenshot_img = kwargs["page_screenshot_img"]
        images = kwargs["images"]

        obs = state_info["observation"][self.obs_modality]
        max_obs_length = self.lm_config.gen_config["max_obs_length"]
        if max_obs_length:
            if self.lm_config.provider == "google":
                print("NOTE: This is a Gemini model, so we use characters instead of tokens for max_obs_length.")
                obs = obs[:max_obs_length]
            else:
                obs = self.tokenizer.decode(self.tokenizer.encode(obs)[:max_obs_length])  # type: ignore[arg-type]

        page = state_info["info"]["page"]
        url = page.url
        previous_action_str = meta_data["action_history"][-1]
        current = template.format(
            objective=intent,
            url=self.map_url_to_real(url),
            observation=obs,
            previous_action=previous_action_str,
        )

        assert all([f"{{k}}" not in current for k in keywords])

        prompt = self.get_lm_api_input(
            intro, examples, current, page_screenshot_img, images
        )
        return prompt

    def get_lm_api_input(
        self,
        intro: str,
        examples: list[InstructionExample],
        current: str,
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> APIInput:
        """Return the require format for an API"""
        examples_ = cast(list[MultimodalExample], examples)
        message: list[dict[str, Any]] | str | list[str | Image.Image]

        if "openai" in self.lm_config.provider:
            match self.lm_config.mode:
                case "chat":
                    message = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": intro}],
                        }
                    ]
                    for example in cast(list[MultimodalChatExample], examples_):
                        example_img = Image.open(example["image_path"])
                        message += [
                            {
                                "role": "user" if "gpt-4o" in self.lm_config.model else "system",
                                "name": "example_user",
                                "content": [
                                    {"type": "text", "text": example["user"]},
                                    {
                                        "type": "text",
                                        "text": "IMAGES: (1) current page screenshot",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": pil_to_b64(example_img)
                                        },
                                    },
                                ],
                            },
                            {
                                "role": "user" if "gpt-4o" in self.lm_config.model else "system",
                                "name": "example_assistant",
                                "content": [{"type": "text", "text": example["assistant"]}],
                            }
                        ]

                    # Encode images and page_screenshot_img as base64 strings.
                    current_prompt = current
                    content: list[dict[str, Any]] = [
                        {
                            "type": "text",
                            "text": "IMAGES: (1) current page screenshot",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": pil_to_b64(page_screenshot_img)},
                        },
                    ]
                    for image_i, image in enumerate(images):
                        content.extend(
                            [
                                {
                                    "type": "text",
                                    "text": f"({image_i+2}) input image {image_i+1}",
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": pil_to_b64(image)},
                                },
                            ]
                        )
                    content = [{"type": "text", "text": current_prompt}] + content

                    message.append({"role": "user", "content": content})
                    return message
                case _:
                    raise ValueError(
                        f"GPT-4V models do not support mode {self.lm_config.mode}"
                    )

        elif "google" in self.lm_config.provider:
            match self.lm_config.mode:
                case "completion":
                    message = cast(list[str | Image.Image], [
                        intro,
                        "Here are a few examples:",
                    ])
                    for example_ in cast(list[MultimodalCompletionExample], examples_):
                        example_img = Image.open(example_["image_path"])
                        message.append(f"Observation\n:{example_['observation']}\n")
                        message.extend(
                            [
                                "IMAGES:",
                                "(1) current page screenshot:",
                                pil_to_vertex(example_img),
                            ]
                        )
                        message.append(f"Action: {example_['action']}")
                    message.append("Now make prediction given the observation")
                    message.append(f"Observation\n:{current}\n")
                    message.extend(
                        [
                            "IMAGES:",
                            "(1) current page screenshot:",
                            pil_to_vertex(page_screenshot_img),
                        ]
                    )
                    for image_i, image in enumerate(images):
                        message.extend(
                            [
                                f"({image_i+2}) input image {image_i+1}",
                                pil_to_vertex(image),
                            ]
                        )
                    message.append("Action:")
                    return message
                case _:
                    raise ValueError(
                        f"Gemini models do not support mode {self.lm_config.mode}"
                    )
        else:
            raise NotImplementedError(
                f"Provider {self.lm_config.provider} not implemented"
            )
