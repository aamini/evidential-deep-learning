import argparse
from collections import defaultdict
import cv2
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
import scipy.stats
import tensorflow as tf
from tqdm import tqdm

import edl
import models

parser = argparse.ArgumentParser()
parser.add_argument("--load-pkl", action='store_true',
                    help="Load predictions for a cached pickle file or \
                        recompute from scratch by feeding the data through \
                        trained models")
args = parser.parse_args()


class Model(Enum):
    GroundTruth = "GroundTruth"
    Dropout = "Dropout"
    Ensemble = "Ensemble"
    Evidential = "Evidential"

save_dir = "pretrained_models"
trained_models = {
    Model.Dropout: [
        "dropout/trial1.h5",
        "dropout/trial2.h5",
        "dropout/trial3.h5",
    ],
    Model.Ensemble: [
        "ensemble/trial1_*.h5",
        "ensemble/trial2_*.h5",
        "ensemble/trial3_*.h5",
    ],
    Model.Evidential: [
        "evidence/trial1.h5",
        "evidence/trial2.h5",
        "evidence/trial3.h5",
    ],
}
output_dir = "figs/depth"

def compute_predictions(batch_size=50, n_adv=9):
    (x_in, y_in), (x_ood, y_ood) = load_data()
    datasets = [(x_in, y_in, False), (x_ood, y_ood, True)]

    df_pred_image = pd.DataFrame(
        columns=["Method", "Model Path", "Input",
            "Target", "Mu", "Sigma", "Adv. Mask", "Epsilon", "OOD"])

    adv_eps = np.linspace(0, 0.04, n_adv)

    for method, model_path_list in trained_models.items():
        for model_i, model_path in enumerate(model_path_list):
            full_path = os.path.join(save_dir, model_path)
            model = models.load_depth_model(full_path, compile=False)

            model_log = defaultdict(list)
            print(f"Running {model_path}")

            for x, y, ood in datasets:
                # max(10,x.shape[0]//500-1)
                for start_i in tqdm(np.arange(0, 3*batch_size, batch_size)):
                    inds = np.arange(start_i, min(start_i+batch_size, x.shape[0]-1))
                    x_batch = x[inds]/np.float32(255.)
                    y_batch = y[inds]/np.float32(255.)

                    if ood:
                        ### Compute predictions and save
                        summary_to_add = get_prediction_summary(
                            method, model_path, model, x_batch, y_batch, ood)
                        df_pred_image = df_pred_image.append(summary_to_add, ignore_index=True)

                    else:
                        ### Compute adversarial mask
                        # mask_batch = create_adversarial_pattern(model, tf.convert_to_tensor(x_batch), tf.convert_to_tensor(y_batch))
                        mask_batch = create_adversarial_pattern(model, x_batch, y_batch)
                        mask_batch = mask_batch.numpy().astype(np.int8)

                        for eps in adv_eps:
                            ### Apply adversarial noise
                            x_batch += (eps * mask_batch.astype(np.float32))
                            x_batch = np.clip(x_batch, 0, 1)

                            ### Compute predictions and save
                            summary_to_add = get_prediction_summary(
                                method, model_path, model, x_batch, y_batch, ood, mask_batch, eps)
                            df_pred_image = df_pred_image.append(summary_to_add, ignore_index=True)



    return df_pred_image


def get_prediction_summary(method, model_path, model, x_batch, y_batch, ood, mask_batch=None, eps=0.0):
    if mask_batch is None:
        mask_batch = np.zeros_like(x_batch)

    ### Collect the predictions
    mu_batch, sigma_batch = predict(method, model, x_batch)
    mu_batch = np.clip(mu_batch, 0, 1)
    sigma_batch = sigma_batch.numpy()

    ### Save the predictions to some dataframes for later analysis
    summary = [{"Method": method.value, "Model Path": model_path,
        "Input": x, "Target": y, "Mu": mu, "Sigma": sigma,
        "Adv. Mask": mask, "Epsilon": eps, "OOD": ood}
        for x,y,mu,sigma,mask in zip(x_batch, y_batch, mu_batch, sigma_batch, mask_batch)]
    return summary


def df_image_to_pixels(df, keys=["Target", "Mu", "Sigma"]):
    required_keys = ["Method", "Model Path"]
    keys = required_keys + keys
    key_types = {key: type(df[key].iloc[0]) for key in keys}
    max_shape = max([np.prod(np.shape(df[key].iloc[0])) for key in keys])

    contents = {}
    for key in keys:
        if np.prod(np.shape(df[key].iloc[0])) == 1:
            contents[key] = np.repeat(df[key], max_shape)
        else:
            contents[key] = np.stack(df[key], axis=0).flatten()

    df_pixel = pd.DataFrame(contents)
    return df_pixel


def gen_cutoff_plot(df_image, eps=0.0, ood=False, plot=True):
    print(f"Generating cutoff plot with eps={eps}, ood={ood}")

    df = df_image[(df_image["Epsilon"]==eps) & (df_image["OOD"]==ood)]
    df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma"])

    df_cutoff = pd.DataFrame(
        columns=["Method", "Model Path", "Percentile", "Error"])

    for method, model_path_list in trained_models.items():
        for model_i, model_path in enumerate(tqdm(model_path_list)):

            df_model = df_pixel[(df_pixel["Method"]==method.value) & (df_pixel["Model Path"]==model_path)]
            df_model = df_model.sort_values("Sigma", ascending=False)
            percentiles = np.arange(100)/100.
            cutoff_inds = (percentiles * df_model.shape[0]).astype(int)

            df_model["Error"] = np.abs(df_model["Mu"] - df_model["Target"])
            mean_error = [df_model[cutoff:]["Error"].mean()
                for cutoff in cutoff_inds]
            df_single_cutoff = pd.DataFrame({'Method': method.value, 'Model Path': model_path,
                'Percentile': percentiles, 'Error': mean_error})

            df_cutoff = df_cutoff.append(df_single_cutoff)

    df_cutoff["Epsilon"] = eps

    if plot:
        print("Plotting cutoffs")
        sns.lineplot(x="Percentile", y="Error", hue="Method", data=df_cutoff)
        plt.savefig(os.path.join(output_dir, f"cutoff_eps-{eps}_ood-{ood}.pdf"))
        plt.show()

        sns.lineplot(x="Percentile", y="Error", hue="Model Path", style="Method", data=df_cutoff)
        plt.savefig(os.path.join(output_dir, f"cutoff_eps-{eps}_ood-{ood}_trial.pdf"))
        plt.show()

        g = sns.FacetGrid(df_cutoff, col="Method", legend_out=False)
        g = g.map_dataframe(sns.lineplot, x="Percentile", y="Error", hue="Model Path")#.add_legend()
        plt.savefig(os.path.join(output_dir, f"cutoff_eps-{eps}_ood-{ood}_trial_panel.pdf"))
        plt.show()


    return df_cutoff


def gen_calibration_plot(df_image, eps=0.0, ood=False, plot=True):
    print(f"Generating calibration plot with eps={eps}, ood={ood}")
    df = df_image[(df_image["Epsilon"]==eps) & (df_image["OOD"]==ood)]
    # df = df.iloc[::10]
    df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma"])

    df_calibration = pd.DataFrame(
        columns=["Method", "Model Path", "Expected Conf.", "Observed Conf."])

    for method, model_path_list in trained_models.items():
        for model_i, model_path in enumerate(tqdm(model_path_list)):

            df_model = df_pixel[(df_pixel["Method"]==method.value) & (df_pixel["Model Path"]==model_path)]
            expected_p = np.arange(41)/40.

            observed_p = []
            for p in expected_p:
                ppf = scipy.stats.norm.ppf(p, loc=df_model["Mu"], scale=df_model["Sigma"])
                obs_p = (df_model["Target"] < ppf).mean()
                observed_p.append(obs_p)

            df_single = pd.DataFrame({'Method': method.value, 'Model Path': model_path,
                'Expected Conf.': expected_p, 'Observed Conf.': observed_p})
            df_calibration = df_calibration.append(df_single)

    df_truth = pd.DataFrame({'Method': Model.GroundTruth.value, 'Model Path': "",
        'Expected Conf.': expected_p, 'Observed Conf.': expected_p})
    df_calibration = df_calibration.append(df_truth)

    df_calibration['Calibration Error'] = np.abs(df_calibration['Expected Conf.'] - df_calibration['Observed Conf.'])
    df_calibration["Epsilon"] = eps
    table = df_calibration.groupby(["Method", "Model Path"])["Calibration Error"].mean().reset_index()
    table = pd.pivot_table(table, values="Calibration Error", index="Method", aggfunc=[np.mean, np.std, scipy.stats.sem])

    if plot:
        print(table)
        table.to_csv(os.path.join(output_dir, "calib_errors.csv"))

        print("Plotting confidence plots")
        sns.lineplot(x="Expected Conf.", y="Observed Conf.", hue="Method", data=df_calibration)
        plt.savefig(os.path.join(output_dir, f"calib_eps-{eps}_ood-{ood}.pdf"))
        plt.show()

        g = sns.FacetGrid(df_calibration, col="Method", legend_out=False)
        g = g.map_dataframe(sns.lineplot, x="Expected Conf.", y="Observed Conf.", hue="Model Path")#.add_legend()
        plt.savefig(os.path.join(output_dir, f"calib_eps-{eps}_ood-{ood}_panel.pdf"))
        plt.show()

    return df_calibration, table



def gen_adv_plots(df_image, ood=False):
    print(f"Generating calibration plot with ood={ood}")
    df = df_image[df_image["OOD"]==ood]
    # df = df.iloc[::10]
    df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma", "Epsilon"])
    df_pixel["Error"] = np.abs(df_pixel["Mu"] - df_pixel["Target"])
    df_pixel["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel["Sigma"]**2))

    ### Plot epsilon vs error per method
    df = df_pixel.groupby([df_pixel.index, "Method", "Model Path", "Epsilon"]).mean().reset_index()
    df_by_method = df_pixel.groupby(["Method", "Model Path", "Epsilon"]).mean().reset_index()
    sns.lineplot(x="Epsilon", y="Error", hue="Method", data=df_by_method)
    plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_method_error.pdf"))
    plt.show()

    ### Plot epsilon vs uncertainty per method
    sns.lineplot(x="Epsilon", y="Sigma", hue="Method", data=df_by_method)
    plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_method_sigma.pdf"))
    plt.show()
    # df_by_method["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_by_method["Sigma"]**2))
    # sns.lineplot(x="Epsilon", y="Entropy", hue="Method", data=df_by_method)
    # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_method_entropy.pdf"))
    # plt.show()


    ### Plot entropy cdf for different epsilons
    df_cumdf = pd.DataFrame(columns=["Method", "Model Path", "Epsilon", "Entropy", "CDF"])
    unc_ = np.linspace(df["Entropy"].min(), df["Entropy"].max(), 100)

    for method in df["Method"].unique():
        for model_path in df["Model Path"].unique():
            for eps in df["Epsilon"].unique():
                df_subset = df[
                    (df["Method"]==method) &
                    (df["Model Path"]==model_path) &
                    (df["Epsilon"]==eps)]
                if len(df_subset) == 0:
                    continue
                unc = np.sort(df_subset["Entropy"])
                prob = np.linspace(0,1,unc.shape[0])
                f_cdf = scipy.interpolate.interp1d(unc, prob, fill_value=(0.,1.), bounds_error=False)
                prob_ = f_cdf(unc_)

                df_single = pd.DataFrame({'Method': method, 'Model Path': model_path,
                    'Epsilon': eps, "Entropy": unc_, 'CDF': prob_})
                df_cumdf = df_cumdf.append(df_single)

    g = sns.FacetGrid(df_cumdf, col="Method")
    g = g.map_dataframe(sns.lineplot, x="Entropy", y="CDF", hue="Epsilon", ci=None).add_legend()
    plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_cdf_method.pdf"))
    plt.show()

    # NOT USED FOR THE FINAL PAPER, BUT FEEL FREE TO UNCOMMENT AND RUN
    # ### Plot calibration for different epsilons/methods
    # print("Computing calibration plots per epsilon")
    # calibrations = []
    # tables = []
    # for eps in tqdm(df["Epsilon"].unique()):
    #     df_calibration, table = gen_calibration_plot(df_image.copy(), eps, plot=False)
    #     calibrations.append(df_calibration)
    #     tables.append(table)
    # df_calibration = pd.concat(calibrations, ignore_index=True)
    # df_table = pd.concat(tables, ignore_index=True)
    # df_table.to_csv(os.path.join(output_dir, f"adv_ood-{ood}_calib_error.csv"))
    #
    #
    # sns.catplot(x="Method", y="Calibration Error", hue="Epsilon", data=df_calibration, kind="bar")
    # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_calib_error_method.pdf"))
    # plt.show()
    #
    # sns.catplot(x="Epsilon", y="Calibration Error", hue="Method", data=df_calibration, kind="bar")
    # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_calib_error_epsilon.pdf"))
    # plt.show()
    #
    # g = sns.FacetGrid(df_calibration, col="Method")
    # g = g.map_dataframe(sns.lineplot, x="Expected Conf.", y="Observed Conf.", hue="Epsilon")
    # g = g.add_legend()
    # plt.savefig(os.path.join(output_dir, f"adv_ood-{ood}_calib_method.pdf"))
    # plt.show()


def gen_ood_comparison(df_image, unc_key="Entropy"):
    print(f"Generating OOD plots with unc_key={unc_key}")

    df = df_image[df_image["Epsilon"]==0.0] # Remove adversarial noise experiments
    # df = df.iloc[::5]
    df_pixel = df_image_to_pixels(df, keys=["Target", "Mu", "Sigma", "OOD"])
    df_pixel["Entropy"] = 0.5*np.log(2*np.pi*np.exp(1.)*(df_pixel["Sigma"]**2))

    df_by_method = df_pixel.groupby(["Method","Model Path", "OOD"])
    df_by_image = df_pixel.groupby([df_pixel.index, "Method","Model Path", "OOD"])

    df_mean_unc = df_by_method[unc_key].mean().reset_index() #mean of all pixels per method
    df_mean_unc_img = df_by_image[unc_key].mean().reset_index() #mean of all pixels in every method and image

    sns.catplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, kind="violin")
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_violin.pdf"))
    plt.show()

    sns.catplot(x="Method", y=unc_key, hue="OOD", data=df_mean_unc_img, kind="box", whis=0.5, showfliers=False)
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_box.pdf"))
    plt.show()


    ### Plot PDF for each Method on both OOD and IN
    g = sns.FacetGrid(df_mean_unc_img, col="Method", hue="OOD")
    g.map(sns.distplot, "Entropy").add_legend()
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_pdf_per_method.pdf"))
    plt.show()


    ### Grab some sample images of most and least uncertainty
    for method in df_mean_unc_img["Method"].unique():
        imgs_max = dict()
        imgs_min = dict()
        for ood in df_mean_unc_img["OOD"].unique():
            df_subset = df_mean_unc_img[
                (df_mean_unc_img["Method"]==method) &
                (df_mean_unc_img["OOD"]==ood)]
            if len(df_subset) == 0:
                continue

            def get_imgs_from_idx(idx):
                i_img = df_subset.loc[idx]["level_0"]
                img_data = df_image.loc[i_img]
                sigma = np.array(img_data["Sigma"])
                entropy = np.log(sigma**2)

                ret = [img_data["Input"], img_data["Mu"], entropy]
                return list(map(trim, ret))

            def idxquantile(s, q=0.5, *args, **kwargs):
                qv = s.quantile(q, *args, **kwargs)
                return (s.sort_values()[::-1] <= qv).idxmax()

            imgs_max[ood] = get_imgs_from_idx(idx=idxquantile(df_subset["Entropy"], 0.95))
            imgs_min[ood] = get_imgs_from_idx(idx=idxquantile(df_subset["Entropy"], 0.05))

        all_entropy_imgs = np.array([ [d[ood][2] for ood in d.keys()] for d in (imgs_max, imgs_min)])
        entropy_bounds = (all_entropy_imgs.min(), all_entropy_imgs.max())

        Path(os.path.join(output_dir, "images")).mkdir(parents=True, exist_ok=True)
        for d in (imgs_max, imgs_min):
            for ood, (x, y, entropy) in d.items():
                id = os.path.join(output_dir, f"images/method_{method}_ood_{ood}_entropy_{entropy.mean()}")
                cv2.imwrite(f"{id}_0.png", 255*x)
                cv2.imwrite(f"{id}_1.png", apply_cmap(y, cmap=cv2.COLORMAP_JET))
                entropy = (entropy - entropy_bounds[0]) / (entropy_bounds[1]-entropy_bounds[0])
                cv2.imwrite(f"{id}_2.png", apply_cmap(entropy))



    ### Plot CDFs for every method on both OOD and IN
    df_cumdf = pd.DataFrame(columns=["Method", "Model Path", "OOD", unc_key, "CDF"])
    unc_ = np.linspace(df_mean_unc_img[unc_key].min(), df_mean_unc_img[unc_key].max(), 200)

    for method in df_mean_unc_img["Method"].unique():
        for model_path in df_mean_unc_img["Model Path"].unique():
            for ood in df_mean_unc_img["OOD"].unique():
                df = df_mean_unc_img[
                    (df_mean_unc_img["Method"]==method) &
                    (df_mean_unc_img["Model Path"]==model_path) &
                    (df_mean_unc_img["OOD"]==ood)]
                if len(df) == 0:
                    continue
                unc = np.sort(df[unc_key])
                prob = np.linspace(0,1,unc.shape[0])
                f_cdf = scipy.interpolate.interp1d(unc, prob, fill_value=(0.,1.), bounds_error=False)
                prob_ = f_cdf(unc_)

                df_single = pd.DataFrame({'Method': method, 'Model Path': model_path,
                    'OOD': ood, unc_key: unc_, 'CDF': prob_})
                df_cumdf = df_cumdf.append(df_single)

    sns.lineplot(data=df_cumdf, x=unc_key, y="CDF", hue="Method", style="OOD")
    plt.savefig(os.path.join(output_dir, f"ood_{unc_key}_cdfs.pdf"))
    plt.show()






def load_data():
    import data_loader
    _, (x_test, y_test) = data_loader.load_depth()
    _, (x_ood_test, y_ood_test) = data_loader.load_apollo()
    print("Loaded data:", x_test.shape, x_ood_test.shape)
    return (x_test, y_test), (x_ood_test, y_ood_test)

def predict(method, model, x, n_samples=10):

    if method == Model.Dropout:
        preds = tf.stack([model(x, training=True) for _ in range(n_samples)], axis=0) #forward pass
        mu, var = tf.nn.moments(preds, axes=0)
        return mu, tf.sqrt(var)

    elif method == Model.Evidential:
        outputs = model(x, training=False)
        mu, v, alpha, beta = tf.split(outputs, 4, axis=-1)
        sigma = tf.sqrt(beta/(v*(alpha-1)))
        return mu, sigma

    elif method == Model.Ensemble:
        # preds = tf.stack([f(x) for f in model], axis=0)
        # y, _ = tf.split(preds, 2, axis=-1)
        # mu = tf.reduce_mean(y, axis=0)
        # sigma = tf.math.reduce_std(y, axis=0)
        preds = tf.stack([f(x) for f in model], axis=0)
        mu, var = tf.nn.moments(preds, 0)
        return mu, tf.sqrt(var)

    else:
        raise ValueError("Unknown model")

def apply_cmap(gray, cmap=cv2.COLORMAP_MAGMA):
    if gray.dtype == np.float32:
        gray = np.clip(255*gray, 0, 255).astype(np.uint8)
    im_color = cv2.applyColorMap(gray, cmap)
    return im_color

def trim(img, k=10):
    return img[k:-k, k:-k]
def normalize(x, t_min=0, t_max=1):
    return ((x-x.min())/(x.max()-x.min())) * (t_max-t_min) + t_min


@tf.function
def create_adversarial_pattern(model, x, y):
    x_ = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        tape.watch(x_)
        if isinstance(model, list):
            preds = tf.stack([model_(x_, training=False) for model_ in model], axis=0) #forward pass
            pred, _ = tf.nn.moments(preds, axes=0)
        else:
            (pred) = model(x_, training=True)
            if pred.shape[-1] == 4:
                pred = tf.split(pred, 4, axis=-1)[0]
        loss = edl.losses.MSE(y, pred)
    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, x_)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad



if args.load_pkl:
    print("Loading!")
    df_image = pd.read_pickle("cached_depth_results.pkl")
else:
    df_image = compute_predictions()
    df_image.to_pickle("cached_depth_results.pkl")


""" ================================================== """
Path(output_dir).mkdir(parents=True, exist_ok=True)
gen_cutoff_plot(df_image)
gen_calibration_plot(df_image)
gen_adv_plots(df_image)
gen_ood_comparison(df_image)
""" ================================================== """
