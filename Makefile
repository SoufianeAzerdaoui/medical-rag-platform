test-backend-contract:
	python -m pytest -q tests/test_chat_api_contract.py -m backend_contract

test-backend:
	python -m pytest -q -m "backend_contract"

