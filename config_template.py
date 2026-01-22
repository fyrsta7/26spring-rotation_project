from pathlib import Path

GITHUB_API_URL = "https://api.github.com"
# GitHub - Settings - Developer Settings - Personal access tokens - Tokens (classic) - 生成一个然后复制进来就行
GITHUB_TOKEN = ""
headers = {"Authorization": GITHUB_TOKEN}

# 该项目根目录（自动获取），注意最后以 "/" 结尾
root_path = str(Path(__file__).parent.resolve()) + "/"

# https://llm.xmcp.ltd/service_portal/
xmcp_base_url = "https://llm.xmcp.ltd"
xmcp_api_key = ""
xmcp_model = "volc/deepseek-v3-250324"
