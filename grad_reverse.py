# merged from https://github.com/janfreyberg/pytorch-revgrad
from torch.autograd import Function
from torch.nn import Module


class GradReverseFunction(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None 
        _, alpha_ = ctx.saved_tensors 
        if ctx.needs_input_grad[0]:
            grad_input = -grad_input * alpha_
        return grad_input, None


class GradReverse(Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """
        A gradient reversal layer.
        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        Potentially scales gradients with scalar alpha
        """
        super().__init__(*args, **kwargs)
        self.alpha = tensor(alpha, requires_grad=False)
        self.apply = GradReverseFunction.apply

    def forward(self, input_):
        return self.apply(input_, self.alpha)