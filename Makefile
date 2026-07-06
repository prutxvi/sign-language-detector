.PHONY: run clean

run:
	python detector.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f *.png *.jpg
