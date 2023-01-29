from setuptools import setup, find_packages

setup(
	name="score_models",
	version="0.1",
	description="Package intended to implement general purpose, "
				"state of the art score model architectures that model continuously "
				"tempered gradient of distributions over images.",
	packages=find_packages(),
	python_requires=">=3.8"
)
