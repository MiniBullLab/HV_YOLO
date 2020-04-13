import torch
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

def summary(model, shape, *args, **kwargs):
    """Summarize the given input model.
    Summarized information are 1) output shape, 2) kernel shape,
    3) number of the parameters and 4) operations (Mult-Adds)
    Args:
        model (Module): Model to summarize
        x (Tensor): Input tensor of the model with [N, C, H, W] shape
                    dtype and device have to match to the model
        args, kwargs: Other argument used in `model.forward` function
    """

    def register_hook(module):
        def hook(module, inputs, outputs):
            cls_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)
            key = "{}_{}".format(module_idx, cls_name)

            info = OrderedDict()
            info["id"] = id(module)
            if isinstance(outputs, (list, tuple)):
                info["out"] = list(outputs[0].size())
            else:
                info["out"] = list(outputs.size())

            info["ksize"] = "-"
            info["stride"] = "-"
            info["inner"] = OrderedDict()
            info["params"], info["macs"] = 0, 0
            info["gradient"] = 0
            for name, param in module.named_parameters():
                info["params"] += param.nelement()
                info["gradient"] = param.requires_grad

                if "weight" == name:
                    ksize = list(param.size())
                    # to make [in_shape, out_shape, ksize, ksize]
                    if len(ksize) > 1:
                        ksize[0], ksize[1] = ksize[1], ksize[0]
                    info["ksize"] = ksize

                    # ignore N, C when calculate Mult-Adds in ConvNd
                    if "Conv" in cls_name:
                        info["macs"] += int(param.nelement() * np.prod(info["out"][2:]))
                    else:
                        info["macs"] += param.nelement()

                    # stride
                    if hasattr(module, 'stride'):
                        info["stride"] = [module.stride[0], module.stride[1]]

                # RNN modules have inner weights such as weight_ih_l0
                elif "weight" in name:
                    info["inner"][name] = list(param.size())
                    info["macs"] += param.nelement()

            # if the current module is already-used, mark as "(recursive)"
            # check if this module has params
            if list(module.named_parameters()):
                for v in summary.values():
                    if info["id"] == v["id"]:
                        info["params"] = "(recursive)"

            if info["params"] == 0:
                info["params"], info["gradient"], info["macs"] = "-", "-", "-"

            summary[key] = info

        # ignore Sequential and ModuleList
        if not module._modules:
            hooks.append(module.register_forward_hook(hook))

    hooks = []
    x = torch.zeros(shape)
    # if model.is_cuda():
    #     input = Variable(x.type(torch.cuda.FloatTensor), requires_grad=False)
    # else:
    input = Variable(x.type(torch.FloatTensor), requires_grad=False)
    summary = OrderedDict({"0_Data":OrderedDict({"id":"0", "ksize":"-", "stride":"-", "out":list(input.shape), "gradient":"False", "params":"_", "macs": "_", \
                                               "inner":OrderedDict()})})

    model.apply(register_hook)
    model(input) if not (kwargs or args) else model(input, *args, **kwargs)

    for hook in hooks:
        hook.remove()

    print("-" * 150)
    print("{:<15} {:>28} {:>15} {:>15} {:>25} {:>20} {:>20}"
          .format("Layer", "Kernel Shape", "Stride", "Gradient", "Output Shape",
                  "# Params (K)", "# Mult-Adds (M)"))
    print("=" * 150)

    total_params, total_macs = 0, 0
    for layer, info in summary.items():
        repr_ksize = str(info["ksize"])
        repr_stride = str(info["stride"])
        repr_out = str(info["out"])
        repr_gradient = str(info["gradient"])
        repr_params = info["params"]
        repr_macs = info["macs"]

        if isinstance(repr_params, (int, float)):
            total_params += repr_params
            repr_params = "{0:,.2f}".format(repr_params / 1000)
        if isinstance(repr_macs, (int, float)):
            total_macs += repr_macs
            repr_macs = "{0:,.2f}".format(repr_macs / 1000000)

        print("{:<15} \t{:>20} {:>15} {:>15} {:>25} {:>20} {:>20}"
              .format(layer, repr_ksize, repr_stride, repr_gradient, repr_out, repr_params, repr_macs))

        # for RNN, describe inner weights (i.e. w_hh, w_ih)
        for inner_name, inner_shape in info["inner"].items():
            print("  {:<13} {:>20}".format(inner_name, str(inner_shape)))

    print("=" * 150)
    print("# Params:    {0:,.2f}K".format(total_params / 1000))
    print("# Mult-Adds: {0:,.2f}M".format(total_macs / 1000000))

    print("-" * 150)

