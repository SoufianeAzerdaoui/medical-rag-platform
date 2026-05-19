# from __future__ import annotations

# import os

# # Centralized runtime defaults for LLM settings.
# # Change values here to affect backend/API + generation/CLI defaults.
# DEFAULT_LLM_PROVIDER = os.getenv("MEDICAL_RAG_LLM_PROVIDER", "ollama")
# # DEFAULT_LLM_MODEL = os.getenv("MEDICAL_RAG_LLM_MODEL", "llama3.2:latest")
# DEFAULT_LLM_MODEL = os.getenv("MEDICAL_RAG_LLM_MODEL", "deepseek-r1:8b")
# DEFAULT_LLM_TEMPERATURE = float(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "0.0"))
# DEFAULT_LLM_NUM_CTX = int(os.getenv("MEDICAL_RAG_LLM_NUM_CTX", "4096"))
# DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "400"))
# DEFAULT_LLM_TIMEOUT = int(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "120"))


from __future__ import annotations

import os

# Centralized runtime defaults for LLM settings.
# Change values here to affect backend/API + generation/CLI defaults.
DEFAULT_LLM_PROVIDER = os.getenv("MEDICAL_RAG_LLM_PROVIDER", "ollama")

# Choix du modèle DeepSeek-R1 8b (Quantifié par défaut par Ollama)
DEFAULT_LLM_MODEL = os.getenv("MEDICAL_RAG_LLM_MODEL", "llama3.2:latest")

# Température à 0.0 pour une précision médicale stricte et logique (Parfait)
DEFAULT_LLM_TEMPERATURE = float(os.getenv("MEDICAL_RAG_LLM_TEMPERATURE", "0.0"))

# [MODIFIÉ] Réduit de 4096 à 2048 pour éviter que le CPU ne sature la RAM de votre PC
DEFAULT_LLM_NUM_CTX = int(os.getenv("MEDICAL_RAG_LLM_NUM_CTX", "2048"))

# [MODIFIÉ] Augmenté de 400 à 1500. DeepSeek-R1 écrit TOUTE sa réflexion <think> en tokens. 
# Si vous laissez 400, la réponse sera coupée au milieu de sa pensée, sans jamais donner le résultat médical.
DEFAULT_LLM_MAX_TOKENS = int(os.getenv("MEDICAL_RAG_LLM_MAX_TOKENS", "400"))

DEFAULT_LLM_TIMEOUT = int(os.getenv("MEDICAL_RAG_LLM_TIMEOUT", "180"))



MEDICAL_RAG_FORCE_LLM_WRITER=1
MEDICAL_RAG_VALIDATE_LLM_FACTS=1
MEDICAL_RAG_LLM_REPAIR_RETRY=1
