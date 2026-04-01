import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ENV_FILE_PATH = os.path.join(PROJECT_ROOT, '.env')
DEFAULT_LLM_NAME = 'HttpsApi'
ENV_LLM_KEY_MAP = {
    'host': 'LLM4AD_API_BASE_URL',
    'key': 'LLM4AD_API_KEY',
    'model': 'LLM4AD_MODEL_ID',
}


def load_env_file(env_path=ENV_FILE_PATH):
    env_vars = {}
    if not os.path.exists(env_path):
        return env_vars

    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue

            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            env_vars[key] = value

    return env_vars


def save_env_file(env_values, env_path=ENV_FILE_PATH):
    existing_lines = []
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as env_file:
            existing_lines = env_file.readlines()

    new_lines = []
    handled_keys = set()

    for raw_line in existing_lines:
        stripped = raw_line.strip()
        if stripped and not stripped.startswith('#') and '=' in raw_line:
            key = raw_line.split('=', 1)[0].strip()
            if key in env_values:
                new_lines.append(f"{key}={env_values[key]}\n")
                handled_keys.add(key)
                continue
        new_lines.append(raw_line)

    if new_lines and not new_lines[-1].endswith('\n'):
        new_lines[-1] += '\n'

    missing_keys = [key for key in env_values if key not in handled_keys]
    if missing_keys and new_lines and new_lines[-1].strip():
        new_lines.append('\n')

    for key in missing_keys:
        new_lines.append(f"{key}={env_values[key]}\n")

    with open(env_path, 'w', encoding='utf-8') as env_file:
        env_file.writelines(new_lines)


def get_saved_llm_parameters():
    env_values = load_env_file()
    saved_parameters = {
        field_name: env_values.get(env_key, '').strip()
        for field_name, env_key in ENV_LLM_KEY_MAP.items()
    }
    saved_parameters['name'] = DEFAULT_LLM_NAME
    return saved_parameters
