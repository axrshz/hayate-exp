"""Run one text request with the default Qwen model."""

from hayate.engine.engine import Engine


def main():
    # The engine loads the model, tokenizer, and CUDA weights before generation.
    engine = Engine("Qwen/Qwen3-4B")
    # The public method returns a Request that stores the decoded response.
    output = engine.generate_text("Explain Artificial General Intelligence")
    print(output.response)


# Run the example only when a user starts this file directly.
if __name__ == "__main__":
    main()
