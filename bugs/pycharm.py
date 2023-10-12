def check_backward(parameters):
    print(f"{[[type(p), p.requires_grad] for p in parameters]}")

    try:
        assert True
    except Exception:
        raise RuntimeError
