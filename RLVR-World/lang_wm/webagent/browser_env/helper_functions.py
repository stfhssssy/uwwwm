import base64
import io
import json
import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from PIL import Image

import numpy.typing as npt
import numpy as np

from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    StateInfo,
    action2str,
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<head>
    <style>
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
    </style>
</head>
<html>
    <body>
     {body}
    </body>
</html>
"""


def get_render_action(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
) -> str:
    """Parse the predicted actions for rendering purpose. More comprehensive information"""
    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["element_id"] in text_meta_data["obs_nodes_info"]:
                node_content = text_meta_data["obs_nodes_info"][
                    action["element_id"]
                ]["text"]
            else:
                node_content = "No match found"
            
            action_str = f'<h2 style="background-color: yellow;">Raw Prediction</h2>'
            action_str += f"<div class='raw_parsed_prediction'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f'<h2 style="background-color: yellow;">Action Object</h2>'
            action_str += f"<div class='action_object'><pre>{repr(action)}</pre></div>"
            action_str += f'<h2 style="background-color: yellow;">Parsed Action</h2>'
            action_str += f"<div class='parsed_action' style='background-color:rgb(231, 131, 151)'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "som":
            meta_data = observation_metadata["image"]
            if action["element_id"] in meta_data["obs_nodes_semantic_info"]:
                node_content = meta_data["obs_nodes_semantic_info"][
                    action["element_id"]
                ]
            else:
                node_content = "No match found"

            action_str = f"<div class='raw_parsed_prediction' style='background-color:grey'><pre>{action['raw_prediction']}</pre></div>"
            action_str += f"<div class='action_object' style='background-color:grey'><pre>{repr(action)}</pre></div>"
            action_str += f"<div class='parsed_action' style='background-color:yellow'><pre>{action2str(action, action_set_tag, node_content)}</pre></div>"

        case "playwright":
            action_str = action["pw_code"]
        case _:
            raise ValueError(
                f"Unknown action type {action['action_type'], action_set_tag}"
            )
    return action_str, action2str(action, action_set_tag, node_content)

def get_render_action_next_state_value_scores(
    action_candidates: list[str],
    next_state_predictions: list[str],
    value_scores: list[float],
    raw_response_for_value_score_calculation: list[list[str]] | None = None,
):
    output_str = '<h2 style="background-color: yellow;">Action Candidate</h2>'
    max_value_scores = max(value_scores)
    for index, (action, prediction, score) in enumerate(zip(action_candidates, next_state_predictions, value_scores)):
        raw_responses = raw_response_for_value_score_calculation[index] if raw_response_for_value_score_calculation else []
        raw_responses_html = "".join([f"<li>{response}</li>" for response in raw_responses])

        output_str += f"""
            <div class="action-candidate-container">
            <h3>Action Candidate {index + 1}</h3>
            <div class="action-candidate">
                <pre>{action}</pre>
            </div>
            <div class="next-state-prediction">
                <pre>{prediction}</pre>
            </div>
            <div class="value-score">
            """
        
        if max_value_scores == score:
            output_str += f'<pre><span style="border: 3px solid red; padding: 2px;">Value score: {score}</span></pre>'
        else:    
            output_str += f'<pre>Value score: {score}</pre>'
        
        output_str += f"""
                    <details>
                        <summary>Toggle Raw Responses</summary>
                        <div class="raw-responses">
                            <h4>Raw Responses:</h4>
                            <ul>
                                {raw_responses_html}
                            </ul>
                        </div>
                    </details>
                </div>
            </div>
            """
        
    return output_str.strip()

def get_action_description(
    action: Action,
    observation_metadata: dict[str, ObservationMetadata],
    action_set_tag: str,
    prompt_constructor: PromptConstructor | None,
) -> str:
    """Generate the text version of the predicted actions to store in action history for prompt use.
    May contain hint information to recover from the failures"""

    match action_set_tag:
        case "id_accessibility_tree":
            text_meta_data = observation_metadata["text"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                if action["element_id"] in text_meta_data["obs_nodes_info"]:
                    node_content = text_meta_data["obs_nodes_info"][
                        action["element_id"]
                    ]["text"]
                    node_content = " ".join(node_content.split()[1:])
                    action_str = action2str(
                        action, action_set_tag, node_content
                    )
                else:
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "som":
            meta_data = observation_metadata["image"]
            if action["action_type"] in [
                ActionTypes.CLICK,
                ActionTypes.HOVER,
                ActionTypes.TYPE,
            ]:
                action_name = str(action["action_type"]).split(".")[1].lower()
                if action["element_id"] in meta_data["obs_nodes_semantic_info"]:
                    node_content = meta_data["obs_nodes_semantic_info"][
                        action["element_id"]
                    ]
                    action_str = action2str(action, action_set_tag, node_content)
                else:
                    action_str = f"Attempt to perfom \"{action_name}\" on element \"[{action['element_id']}]\" but no matching element found. Please check the observation more carefully."
            else:
                if (
                    action["action_type"] == ActionTypes.NONE
                    and prompt_constructor is not None
                ):
                    action_splitter = prompt_constructor.instruction[
                        "meta_data"
                    ]["action_splitter"]
                    action_str = f'The previous prediction you issued was "{action["raw_prediction"]}". However, the format was incorrect. Ensure that the action is wrapped inside a pair of {action_splitter} and enclose arguments within [] as follows: {action_splitter}action [arg] ...{action_splitter}.'
                else:
                    action_str = action2str(action, action_set_tag, "")

        case "playwright":
            action_str = action["pw_code"]

        case _:
            raise ValueError(f"Unknown action type {action['action_type']}")

    return action_str


class RenderHelper(object):
    """Helper class to render text and image observations and meta data in the trajectory"""

    def __init__(
        self, config_file: str, result_dir: str, action_set_tag: str, prefix: str = ""
    ) -> None:
        with open(config_file, "r") as f:
            _config = json.load(f)
            _config_str = ""
            self.reference_answer = None
            for k, v in _config.items():
                _config_str += f"{k}: {v}\n"
                if k == 'eval':
                    if v['eval_types'][0] == 'string_match':
                        self.reference_answer = v['reference_answers'] # exact_match -> str or must_include -> list or fuzzy_match -> list
                    elif v['eval_types'][0] == 'url_match':
                        self.reference_answer = v['reference_url'] # -> str
            _config_str = f"<pre>{_config_str}</pre>\n"
            task_id = _config["task_id"]

        self.action_set_tag = action_set_tag

        if prefix:
            self.render_file = open(
                Path(result_dir) / f"render_{prefix}_{task_id}.html", "a+", encoding="utf-8"
            )
        else:
            self.render_file = open(
                Path(result_dir) / f"render_{task_id}.html", "a+", encoding="utf-8"
            )
        self.render_file.truncate(0)
        # write init template
        self.render_file.write(HTML_TEMPLATE.format(body=f"{_config_str}"))
        self.render_file.read()
        self.render_file.flush()

    def render(
        self,
        action: Action,
        state_info: StateInfo,
        meta_data: dict[str, Any],
        render_screenshot: bool = False,
        additional_text: list[str] | None = None,
        action_candidates: list[str] | None = None,
        next_state_predictions: list[str] | None = None,
        value_scores: list[float] | None = None,
        raw_response_for_value_score_calculation: str | None = None,
    ) -> None:
        """Render the trajectory"""
        # text observation
        observation = state_info["observation"]
        text_obs = observation["text"]
        info = state_info["info"]
        new_content = f"<hr style=\"border: none; height: 10px; background-color: black;\">"
        new_content += f"<h1>New Page</h1>\n"
        new_content += f'<h2 style="background-color: yellow;">URL</h2>'
        new_content += f"<h3 class='url'><a href={state_info['info']['page'].url}>{state_info['info']['page'].url}</a></h3>\n"
        new_content += f"<h2 style=\"background-color: yellow;\">Observation</h2>"
        new_content += f"<div class='state_obv'><pre>{text_obs}</pre><div>\n"

        if render_screenshot:
            # image observation
            img_obs: npt.NDArray[np.uint8] = observation["image"] # type: ignore[assignment]
            image = Image.fromarray(img_obs)
            byte_io = io.BytesIO()
            image.save(byte_io, format="PNG")
            byte_io.seek(0)
            image_bytes = base64.b64encode(byte_io.read())
            image_str = image_bytes.decode("utf-8")
            new_content += f"<h2 style=\"background-color: yellow;\">Image Screenshot</h2>"
            new_content += f"<div style=\"text-align: center;\"><img src='data:image/png;base64,{image_str}' style='width:80vw; height:auto;'/></div>\n"

        # meta data
        new_content += f"<h2 style=\"background-color: yellow;\">Action History</h2>"
        action_history = '\n'.join(meta_data['action_history'])
        new_content += f"<div class='prev_action'>{action_history}</div>\n"
        new_content += f"<h2 style=\"background-color: yellow;\">Previous Action</h2>"
        new_content += f"<div class='prev_action'>{meta_data['action_history'][-1]}</div>\n"

        # additional text
        if additional_text:
            for text_i, text in enumerate(additional_text):
                # Alternate background color between light green and light blue
                if text_i % 2 == 0:
                    bg_color = "#87CEFA"
                else:
                    bg_color = "#98FB98"
                new_content += f"<div class='additional_text' style='background-color: {bg_color}'>#{text_i+1}: {text}</div>\n"

        # action
        action_str, action2str = get_render_action(
            action,
            info["observation_metadata"],
            action_set_tag=self.action_set_tag,
        )
        new_content += f"{action_str}\n"

        if action_candidates:
            next_state_predictions_str = get_render_action_next_state_value_scores(
                action_candidates=action_candidates,
                next_state_predictions=next_state_predictions,
                value_scores=value_scores,
                raw_response_for_value_score_calculation=raw_response_for_value_score_calculation
            )
        new_content += f"{next_state_predictions_str}\n"

        # Reference answers
        new_content += f"<h2 style=\"background-color: yellow;\">Reference Answers</h2>"
        new_content += f"<div class='reference'><pre>{self.reference_answer}</pre><div>\n"

        # add new content
        self.render_file.seek(0)
        html = self.render_file.read()
        soup = BeautifulSoup(html, 'html.parser')
        body_tag = soup.body

        if body_tag:
            html_body = str(body_tag)
            html_body += new_content
        else:
            # Handle the case when <body> tag is not found
            html_body = new_content

        html = HTML_TEMPLATE.format(body=html_body)
        self.render_file.seek(0)
        self.render_file.truncate()
        self.render_file.write(html)
        self.render_file.flush()

    def extract_results(self, text):
        # TODO: Check for other patterns
        pattern = r'^stop \[(.*?)\]'

        match = re.search(pattern, text)
        if match:
            return match.group(1)
        else:
            return "No match found"

    def close(self) -> None:
        self.render_file.close()
