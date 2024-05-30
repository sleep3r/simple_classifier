# Set default variables
PACKAGE_NAME ?= simple_classifier

# Declare phony targets
.PHONY: lint format test train

# Linting
lint:
	@mypy $(PACKAGE_NAME)
	@ruff $(PACKAGE_NAME)

# Formatting
format:
	@isort $(PACKAGE_NAME)
	@black $(PACKAGE_NAME)


# Run train
train:
	python simple_classifier/train.py --config $(CONFIG)
