import evaluate as hf_evaluate


try:
    compute_ = hf_evaluate.load("code_eval")
    test_cases = ["assert add(2, 3)==5"]
    candidates = [["def add(a,b): return a*b"]]
    results = compute_.compute(references=test_cases, predictions=candidates, k=[1])
except Exception as e:
    raise e


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    global compute_
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = compute_.compute(
        references=references,
        predictions=predictions,
        k=k
    )
    return res[0]


def clean_response_string(r: str) -> str:
    cleaned_text = r if r.rfind("```python") == -1 else r[r.rfind("```python"):]
    cleaned_text = cleaned_text if cleaned_text.rfind("```") == -1 else cleaned_text[: cleaned_text.rfind("```")]
    cleaned_text = cleaned_text if cleaned_text.rfind("if __name__ == \"__main__\":") == -1 else cleaned_text[: cleaned_text.rfind("if __name__ == \"__main__\":")]
    return cleaned_text

    
def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [clean_response_string(r) for r in resp]
        for resp, doc in zip(resps, docs)
    ]
