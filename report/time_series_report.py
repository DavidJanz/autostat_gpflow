import kernels


def kernel_description(model):
    model.simplify()  # type: kernels.kernels_abstract.KernelWrapper
    kernel_components = model.break_into_summands()
    print(kernel_components)

    print([(k, p.read_value()) for k in kernel_components for p in k.gpf_kernel.parameters])
    # todo: translate order_by_mae_reduction
    # todo: translate component_stats_and_plots
    # todo: translate checking_stats

    return {}
