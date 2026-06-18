.PHONY: install test demo dashboard train clean

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest

demo:
	python3 -m feathersim.demo

dashboard:
	python3 -m uvicorn feathersim.dashboard.server:app --reload

train:
	python3 -m feathersim.perception.train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
