import sys
import types
from pathlib import Path

APPLICATION_DIR = Path(__file__).resolve().parents[1]
if str(APPLICATION_DIR) not in sys.path:
    sys.path.append(str(APPLICATION_DIR))

# Stub optional dependencies so config.py can be imported in a lightweight unit test.
dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

prompts_module = types.ModuleType("langchain_core.prompts")

class _FakeChatPromptTemplate:
    @classmethod
    def from_messages(cls, messages):
        return {"messages": messages}

prompts_module.ChatPromptTemplate = _FakeChatPromptTemplate
sys.modules.setdefault("langchain_core.prompts", prompts_module)

from config import build_qa_system_prompt


def test_build_qa_system_prompt_injects_business_unit_and_preserves_context_placeholder():
    prompt = build_qa_system_prompt("Plantations")

    assert "KPN BU Plantations" in prompt
    assert "{context}" in prompt


def test_build_qa_system_prompt_defaults_to_kpn_corp_when_business_unit_missing():
    prompt = build_qa_system_prompt(None)

    assert "Assistant Human Capital Officer di KPN Corp" in prompt
