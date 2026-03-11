.PHONY: all clean

PYTHON_FILES := $(wildcard *.py)
TARGET := result

all: $(TARGET)

$(TARGET): $(PYTHON_FILES)
	python -m py_compile $(PYTHON_FILES)
	python -m zipapp -c -o $(TARGET) .

clean:
	rm -f $(TARGET)
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete