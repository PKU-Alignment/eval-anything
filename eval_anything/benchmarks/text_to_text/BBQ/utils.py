def check_eval_response(eval_response: str) -> int:
    if response.startswith('0'):
        return 0
    elif response.startswith('1'):
        return 1
    elif response.startswith('2'):
        return 2
    else:
        raise ValueError(f"Invalid eval response: {eval_response}")
