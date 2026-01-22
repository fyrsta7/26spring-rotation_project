import openai
import config

client = openai.OpenAI(
    api_key = config.xmcp_api_key,
    base_url = config.xmcp_base_url # LiteLLM Proxy is OpenAI compatible, Read More: https://docs.litellm.ai/docs/proxy/user_keys
)

response = client.chat.completions.create(
    model = config.xmcp_model, # model to send to the proxy
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ],
    temperature = 0.3,
    max_tokens = 1024
)

print(response)

print("\n\n")

print(response.choices[0].message.content)