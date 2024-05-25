from infra.evaluator.evaluator import Evaluator

__all__ = ["evaluate_latency", "evaluate_accuracy"]

def evaluate_latency(model, input_shape, backend, command):
    """
    Evaluate the latency of the model on the device
    After evaluation, the model will be sent to cuda
    """
    evaluator = Evaluator()
    latency = evaluator.evaluate_latency(
        model,
        input_shape,
        f"configs/backends/{backend}.json",
        f"configs/commands/{command}.json",
    )
    model.cuda()
    return latency.avg / 1000  # convert to ms

def evaluate_accuracy(model, evaluate):
    evaluator = Evaluator(
        f"configs/evaluate/{evaluate}.json"
    )
    acc = evaluator.evaluate_accuracy(model)
    model.cuda()
    return acc