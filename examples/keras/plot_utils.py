def get_roc_curve(y_pred, y_test, w_test=None):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred, sample_weight=w_test)
    return fpr, tpr


def plot_roc_curve(pred, target, figure_name="tmp.png"):

    import matplotlib.pyplot as plt
    fpr, tpr = get_roc_curve(pred, target)

    from sklearn.metrics import auc
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr,
             tpr,
             lw=1,
             alpha=0.3,
             label=f'ROC fold (AUC = {roc_auc :0.3f})')
    plt.legend()
    plt.xlabel('False positive (BG efficiency)')
    plt.ylabel('True positive (Sig efficiency)')
    plt.savefig(figure_name)
    plt.close()


def plot_regression_pull(pred, target, variables, save_dir="tmp"):
    res = {}
    for i, var in enumerate(variables):
        res[var] = pred[:, i] - target[:, i]

    import numpy as np
    vmin = min([np.quantile(v, 0.01) for v in res.values()])
    vmax = max([np.quantile(v, 0.99) for v in res.values()])
    import matplotlib.pyplot as plt
    arg = {'bins': 100, 'histtype': 'step', 'density': True}
    for var, value in res.items():
        plt.hist(value, range=(vmin, vmax), **arg)
        plt.savefig(f"{save_dir}/{var}.png", dpi=300)
        plt.close()


def plot_classification(storegate,
                        var_pred=[],
                        var_target=[],
                        data_id="",
                        phase="test",
                        save_dir="tmp"):
    storegate.set_data_id(data_id)
    y_test = storegate.get_data(phase=phase,
                                var_names=var_target)
    y_pred = storegate.get_data(phase=phase,
                                var_names=var_pred)

    fpr, tpr = get_roc_curve(y_pred, y_test)
    with open(f"{save_dir}/fpr_tpr.pkl", 'wb') as f:
        import pickle
        pickle.dump({"fpr": fpr, "tpr": tpr}, f)

    plot_roc_curve(y_pred, y_test, figure_name=save_dir + "/roc.png")


def plot_regression(storegate,
                    var_pred=[],
                    var_target=[],
                    data_id="",
                    phase="test",
                    save_dir="tmp"):
    storegate.set_data_id(data_id)
    y_test = storegate.get_data(phase=phase,
                                var_names=var_target)
    y_pred = storegate.get_data(phase=phase,
                                var_names=var_pred)

    from multiml_htautau.task.loss import Tau4vecCalibLoss_np
    mse = Tau4vecCalibLoss_np(pt_scale=1e-2, use_pxyz=True)(y_test, y_pred)
    if phase is None:
        phase = "total"
    from multiml import logger
    logger.debug(f"mse ({phase}) = {mse}")
    outputname = f"{save_dir}/mse.{phase}.pkl"
    with open(outputname, 'wb') as f:
        import pickle
        pickle.dump({"mse": mse}, f)

    plot_regression_pull(y_pred, y_test, var_target, save_dir)


def plot_system_mass(storegate,
                     var_pred=[],
                     label="",
                     data_id="",
                     phase="test",
                     save_dir="tmp"):
    storegate.set_data_id(data_id)
    y_pred = storegate.get_data(phase=phase,
                                var_names=var_pred)
    label = storegate.get_data(phase=phase, var_names=label)
    is_sig = (label == 1.)
    is_bkg = (label == 0.)

    import numpy as np
    y_pred = y_pred.reshape(-1, 3)

    def _convert(x):
        pt = np.exp(x[:, 0]) - 0.1
        eta = x[:, 1]
        phi = x[:, 2]
        mass = 1.777  # x[:, 3]
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
        px = np.expand_dims(px, -1)
        py = np.expand_dims(py, -1)
        pz = np.expand_dims(pz, -1)
        e = np.expand_dims(e, -1)
        return np.concatenate([px, py, pz, e], axis=1)

    def _get_mass(x):
        px = x[:, 0]
        py = x[:, 1]
        pz = x[:, 2]
        e = x[:, 3]
        return np.sqrt(e**2 - (px**2 + py**2 + pz**2))

    y_pred = _convert(y_pred).reshape(-1, 2, 4)
    y_pred = np.sum(y_pred, axis=1).reshape([-1, 4])
    y_pred = _get_mass(y_pred)

    import matplotlib.pyplot as plt
    arg = {'bins': 100, 'histtype': 'step', 'density': True}
    plt.hist(y_pred[is_sig], range=(0., 200.), **arg, label='Signal')
    plt.hist(y_pred[is_bkg], range=(0., 200.), **arg, label='Background')
    plt.legend()
    plt.savefig(f"{save_dir}/mass_pred.png", dpi=300)
    plt.close()


def dump_predictions(storegate,
                     variables=[],
                     data_id="",
                     phase="test",
                     save_dir="tmp"):

    storegate.set_data_id(data_id)
    y_pred = storegate.get_data(phase=phase,
                                var_names=variables)

    import os
    import numpy as np
    os.makedirs(f"{save_dir}/prediction/{phase}", exist_ok=True)

    if len(y_pred.shape) == 1:
        y_pred = y_pred.reshape([-1, 1])

    for i, v in enumerate(variables):
        np.save(f"{save_dir}/prediction/{phase}/{v}", y_pred[:, i])
