import os
import json
import time
import requests
import openai
import copy
import together
from together import AsyncTogether, Together
import datetime

import time
import google.generativeai as genai
import anthropic

from loguru import logger
# from vertexai.preview.generative_models import GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold, SafetySetting
import base64

safe = [
        {
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ]

os.environ['TOGETHER_API_KEY'] = 'Your API key'
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
os.environ['OPENAI_API_KEY'] = 'Your API key'
os.environ['GEMINI_API_KEY'] = 'Your API key'
os.environ['ANTHROPIC_API_KEY'] = 'Your API key'

DEBUG = int(os.environ.get("DEBUG", "0"))



def generate_together(
    model,
    messages,
    temperature=0.0,
    streaming=False,
    n=1,
):

    output = None
    # messages = [{"role": "user", "content": messages}]

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                    "n":n,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = [item["message"]["content"] for item in res.json()["choices"]]
            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    # output = output.strip()
    output = [item.strip() for item in output][0]

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output



def generate_openai(
    model,
    messages,
    temperature=0.0,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_gemini(
    model,
    messages,
    temperature=0.0
):
    try:
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        model = genai.GenerativeModel(model)
        response = model.generate_content(messages, safety_settings=safe)
        return response.text.strip()
    except Exception as E:
        print(E)
        return None


def generate_vllm(messages, model, n=1):

    client = openai.OpenAI(api_key="EMPTY", base_url=os.environ.get("base_url"))
    kwargs = {
        'model': model,
        'messages': messages,
        "n":n
    }
    completion = client.chat.completions.create(**kwargs)
    res = [item.message.content.strip() for item in completion.choices][0]
    return res


def generate_claude(
    messages,
    model="claude-3-5-sonnet-20241022",
    temperature=0.0,
    max_tokens=4096
):
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    # messages = [{"role": "user", "content": messages}]
    output = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **messages
    )
    return output.content[0].text.strip()


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



import ast

def extract_dict(s):
    s = s.replace("'","").replace("\n","")
    in_string = None  # None, 'single', 'double'
    escape = False
    brace_balance = 0
    start_index = -1

    for i, c in enumerate(s):
        if in_string is None:
            if c == '{':
                if brace_balance == 0:
                    start_index = i
                brace_balance += 1
            elif c == '}':
                if brace_balance > 0:
                    brace_balance -= 1
                    if brace_balance == 0 and start_index != -1:
                        candidate = s[start_index:i+1]
                        # Replace each backslash with two to preserve them after parsing
                        modified_candidate = candidate.replace('\\', '\\\\')
                        try:
                            parsed = ast.literal_eval(modified_candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except (SyntaxError, ValueError):
                            # Parsing failed, reset and continue searching
                            pass
                        start_index = -1
                        brace_balance = 0
            elif c in ('"', "'") and not escape:
                in_string = c
        else:
            # Inside a string
            if escape:
                escape = False
            elif c == '\\':
                escape = True
            elif c == in_string and not escape:
                in_string = None

        # Reset escape if the current character is not a backslash
        if c != '\\':
            escape = False

    return None


def build_message(model, setting, system_prompt, user_prompt, image_path):
    """
    Constructs a message payload compatible with different AI models and settings.

    Parameters:
    - model (str): The name of the AI model (e.g., 'gpt-4o', 'claude-3-opus', 'gemini-pro').
    - setting (str): A string indicating settings; if it contains 'V', vision capabilities are enabled.
    - system_prompt (str): The system prompt to guide the AI's behavior.
    - user_prompt (str): The user prompt related to the task.
    - image_path (str): The file path to the local image.

    Returns:
    - dict or list: A dictionary or list representing the message payload, depending on the API requirements.
    """

    # Determine if vision capabilities are enabled
    vision_enabled = "V" in setting.upper()

    # Determine the API type based on the model name
    model_lower = model.lower()
    if "claude" in model_lower:
        api_type = "anthropic"
    elif "gemini" in model_lower:
        api_type = "gemini"
    elif "gpt" in model_lower:
        api_type = "openai"
    else:
        api_type = "together-vllm"

    # Initialize variables
    base64_image = None
    mime_type = None

    # If vision is enabled, read and encode the image
    if vision_enabled and image_path:
        # Determine the MIME type based on the file extension
        ext = os.path.splitext(image_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext == ".png":
            mime_type = "image/png"
        elif ext == ".webp":
            mime_type = "image/webp"
        elif ext == ".gif":
            mime_type = "image/gif"
        elif ext in [".heic", ".heif"]:
            mime_type = "image/heic"
        else:
            raise ValueError(f"Unsupported image format: {ext}")

        # Read and encode the image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the message payload based on the API type
    if api_type == "anthropic":
        # For Anthropic Claude models
        content = []
        if vision_enabled and base64_image:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_image
                }
            })
        content.append({
            "type": "text",
            "text": user_prompt
        })
        messages = {
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

    elif api_type == "gemini":
        # For Google Gemini models
        parts = []
        if vision_enabled and base64_image:
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_image
                }
            })
        parts.append({
            "text": user_prompt
        })
        messages = [
            {
                "role": "user",
                "parts": parts
            }
        ]

    elif api_type == "together-vllm":
        # For Together.ai and vLLM models
        user_content = []
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        if vision_enabled and base64_image:
            image_data_uri = f"data:{mime_type};base64,{base64_image}"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_data_uri
                }
            })
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

    else:
        # Default to OpenAI API format
        user_content = []
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
        if vision_enabled and base64_image:
            image_data_uri = f"data:{mime_type};base64,{base64_image}"
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_data_uri
                }
            })
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]

    return messages



model2function = {
    "gpt-4o-2024-08-06": generate_openai,
    "gpt-4.1-mini-2025-04-14": generate_openai,
    "claude-3-7-sonnet-20250219": generate_claude,
    "claude-3-5-haiku-20241022": generate_claude,
    "gemini-2.5-pro-exp-03-25": generate_gemini,
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": generate_together,
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": generate_together,
    "Qwen/Qwen2.5-VL-72B-Instruct": generate_vllm,
    "Qwen/Qwen2.5-VL-3B-Instruct": generate_vllm,
    "Qwen/Qwen2.5-VL-7B-Instruct": generate_vllm,
    "Qwen/Qwen2.5-VL-32B-Instruct": generate_vllm
}
