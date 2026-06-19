.PHONY: install test demo fleet dashboard teleop train policy clean

install:
	python3 -m pip install -r requirements.txt

test:
	python3 -m pytest

demo:
	python3 -m feathersim.demo

fleet:
	python3 -m feathersim.fleet.demo

# No --reload: the sim is a stateful in-process thread; a reload would reset the running demo.
# `dashboard` = the v2 multi-robot command center; `teleop` = the single-robot WASD-override dashboard.
dashboard:
	python3 -m uvicorn feathersim.dashboard.fleet_server:app --port 8000

teleop:
	python3 -m uvicorn feathersim.dashboard.server:app --port 8000

train:
	python3 -m feathersim.perception.train

policy:
	python3 -m feathersim.policy.train

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache
