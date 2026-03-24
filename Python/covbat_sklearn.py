"""
CovBat sklearn-style wrapper
============================
Wraps the covbat/combat functions from kjamison/CovBat_Harmonization
(forked from andy1764/CovBat_Harmonization) with a scikit-learn-compatible
fit() / transform() interface.

The original covbat() function estimates and applies harmonization in one
pass with no way to reuse parameters on new data. This wrapper splits those
two phases by:

  1. Running the covbat estimation on training data and capturing all
     intermediate parameters (grand_mean, var_pooled, gamma_star/delta_star,
     PCA loadings, StandardScaler, and ComBat-on-scores parameters).
  2. Applying those frozen parameters to new data in transform() without
     re-estimating anything from the test set.

Usage
-----
    from covbat_sklearn import CovBatHarmonizer
    import pandas as pd

    # data: pd.DataFrame of shape (n_features, n_samples)
    harmonizer = CovBatHarmonizer(pct_var=0.95)
    train_harmonized = harmonizer.fit_transform(data_train, batch_train)
    test_harmonized  = harmonizer.transform(data_test, batch_test)

    # with biological covariates (model is a DataFrame of shape (n_samples, n_covariates)):
    harmonizer = CovBatHarmonizer(pct_var=0.95, numerical_covariates="age")
    harmonizer.fit_transform(data_train, batch_train, model=model_train)
    harmonizer.transform(data_test, batch_test, model=model_test)

    #example with trainidx and testidx arrays for splitting data:
    trainidx = np.random.choice(data.columns, size=int(0.8*data.shape[1]), replace=False)
    testidx = np.array([c for c in data.columns if c not in trainidx])
    harmonizer = CovBatHarmonizer(pct_var=0.95, numerical_covariates="age")
    train_harmonized = harmonizer.fit_transform(data.loc[:, trainidx], batch.loc[trainidx], model=model.loc[trainidx])
    test_harmonized = harmonizer.transform(data.loc[:, testidx], batch.loc[testidx], model=model.loc[testidx])

Data conventions (identical to original covbat.py)
---------------------------------------------------
    data  : pd.DataFrame, shape (n_features, n_samples)
    batch : pd.Series or array-like, length n_samples, index matching data.columns
    model : pd.DataFrame or None, shape (n_samples, n_covariates)
            biological covariates to preserve; do NOT include a batch column

Serialization
-------------
    import pickle
    with open("harmonizer.pkl", "wb") as f:
        pickle.dump(harmonizer, f)
    harmonizer2 = pickle.load(open("harmonizer.pkl", "rb"))
    test_harmonized = harmonizer2.transform(data_test, batch_test)
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import numpy.linalg as la
import pandas as pd
import patsy

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Import helpers from covbat.py (we reuse, not re-implement)
# ---------------------------------------------------------------------------
try:
    from covbat import design_mat, it_sol, aprior, bprior
except ImportError:
    try:
        from Python.covbat import design_mat, it_sol, aprior, bprior
    except ImportError as exc:
        raise ImportError(
            "Could not import helper functions from covbat.py. "
            "Make sure covbat.py is on sys.path."
        ) from exc


# ---------------------------------------------------------------------------
# _covbat_fit: estimate all parameters + return corrected training data
# ---------------------------------------------------------------------------

def _covbat_fit(data: pd.DataFrame, batch, model=None,
                numerical_covariates=None, pct_var: float = 0.95, n_pc: int = 0):
    """
    Full covbat estimation on training data.
    Returns (corrected_data: pd.DataFrame, params: dict).
    params holds every value needed to apply correction to unseen data.
    """
    if isinstance(numerical_covariates, str):
        numerical_covariates = [numerical_covariates]
    if numerical_covariates is None:
        numerical_covariates = []

    if model is not None and isinstance(model, pd.DataFrame):
        model = model.copy()
        model["batch"] = list(batch)
    else:
        model = pd.DataFrame({"batch": batch})

    batch_items  = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info   = [v for k, v in batch_items]
    n_batch      = len(batch_info)
    n_batches    = np.array([len(v) for v in batch_info])
    n_array      = float(sum(n_batches))

    data_constant_tol=1e-8
    feat_mask=(data.std(axis=1) > data_constant_tol).to_numpy()
    
    data_orig_shape=data.shape
    data_removed=data[~feat_mask]
    data = data[feat_mask]
    if not feat_mask.all():
        print(f"Removing {(~feat_mask).sum()}/{feat_mask.shape[0]} constant-valued features")
    
    # drop intercept columns
    drop_cols = [cname for cname, inter in ((model == 1).all()).items() if inter]
    model = model[[c for c in model.columns if c not in drop_cols]]
    numerical_covariates = [
        list(model.columns).index(c) if isinstance(c, str) else c
        for c in numerical_covariates if c not in drop_cols
    ]

    design = design_mat(model, numerical_covariates, batch_levels)

    # ── ComBat: standardize ─────────────────────────────────────────────────
    B_hat      = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), data.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch, :])
    var_pooled = np.dot(
        ((data - np.dot(design, B_hat).T) ** 2),
        np.ones((int(n_array), 1)) / int(n_array),
    )
    stand_mean = np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array)))
    )
    tmp = np.array(design.copy())
    tmp[:, :n_batch] = 0
    stand_mean += np.dot(tmp, B_hat).T

    s_data = (data - stand_mean) / np.dot(
        np.sqrt(var_pooled), np.ones((1, int(n_array)))
    )

    # ── ComBat: estimate batch effects ──────────────────────────────────────
    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(
        np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T),
        s_data.T,
    )
    delta_hat = [s_data[bi].var(axis=1) for bi in batch_info]

    gamma_bar = gamma_hat.mean(axis=1)
    t2        = gamma_hat.var(axis=1)
    a_prior   = list(map(aprior, delta_hat))
    b_prior   = list(map(bprior, delta_hat))

    gamma_star, delta_star = [], []
    for i, batch_idxs in enumerate(batch_info):
        temp = it_sol(
            s_data[batch_idxs], gamma_hat[i], delta_hat[i],
            gamma_bar[i], t2[i], a_prior[i], b_prior[i],
        )
        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)

    # ── ComBat: apply correction ─────────────────────────────────────────────
    bayesdata = s_data.copy()
    for j, batch_idxs in enumerate(batch_info):
        dsq   = np.sqrt(delta_star[j, :]).reshape((-1, 1))
        denom = np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(
            bayesdata[batch_idxs]
            - np.dot(batch_design.loc[batch_idxs], gamma_star).T
        )
        bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    # NOTE: stand_mean is NOT added back here (matches original covbat.py)
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

    # ── CovBat: PCA on standardized residuals ───────────────────────────────
    comdata = bayesdata.T.values           # (n_samples, n_features)
    scaler  = StandardScaler()
    comdata_scaled = scaler.fit_transform(comdata)

    pca = PCA()
    pca.fit(comdata_scaled)

    full_scores_arr = pca.transform(comdata_scaled)   # (n_samples, n_pcs)
    full_scores     = pd.DataFrame(full_scores_arr.T) # (n_pcs, n_samples)
    full_scores.columns = data.columns

    var_exp = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4))
    npc = int(np.min(np.where(var_exp > pct_var))) + 1
    if n_pc > 0:
        npc = n_pc

    final_pct_var=np.round(var_exp[npc - 1],decimals=4)
    # ── CovBat: ComBat (no EB) on the leading PC scores ─────────────────────
    scores = full_scores.loc[range(npc), :]
    scores_corrected, score_params = _combat_scores_fit(scores, batch)
    full_scores.loc[range(npc), :] = scores_corrected

    # ── Reconstruct data ─────────────────────────────────────────────────────
    x_covbat = bayesdata - bayesdata   # zeros, preserving DataFrame structure
    proj = np.dot(full_scores.T.values, pca.components_)   # (n_samples, n_features)
    x_covbat += scaler.inverse_transform(proj).T
    x_covbat += stand_mean

    if not feat_mask.all():
        #if we masked out constant features, copy those features back into the final output
        x_covbat_full=pd.DataFrame(
            np.zeros(data_orig_shape),
            columns=data.columns
        )
        x_covbat_full.loc[feat_mask]=x_covbat
        x_covbat_full.loc[~feat_mask]=data_removed
        x_covbat=x_covbat_full
    
    params = dict(
        # ComBat parameters
        batch_levels         = batch_levels,
        batch_info           = batch_info,
        n_batch              = n_batch,
        n_batches            = n_batches,
        B_hat                = B_hat,       # needed to reconstruct stand_mean for new data
        grand_mean           = grand_mean,
        var_pooled           = var_pooled,
        gamma_star           = gamma_star,
        delta_star           = delta_star,
        # PCA / scaler
        scaler               = scaler,
        pca                  = pca,
        npc                  = npc,
        final_pct_var        = final_pct_var,
        # Score ComBat parameters
        score_params         = score_params,
        # Metadata for rebuilding design matrix on new data
        drop_cols            = drop_cols,
        numerical_covariates = numerical_covariates,
        feat_mask            = feat_mask,
    )
    return x_covbat, params


# ---------------------------------------------------------------------------
# _combat_scores_fit: ComBat (eb=False) on PC scores, return params
# ---------------------------------------------------------------------------

def _combat_scores_fit(scores: pd.DataFrame, batch):
    """Fit ComBat (eb=False) on PC scores. Returns (corrected, params)."""
    model = pd.DataFrame({"batch": batch})
    batch_items  = model.groupby("batch").groups.items()
    batch_levels = [k for k, v in batch_items]
    batch_info   = [v for k, v in batch_items]
    n_batch      = len(batch_info)
    n_batches    = np.array([len(v) for v in batch_info])
    n_array      = float(sum(n_batches))

    design = design_mat(model, [], batch_levels)

    B_hat      = np.dot(np.dot(la.inv(np.dot(design.T, design)), design.T), scores.T)
    grand_mean = np.dot((n_batches / n_array).T, B_hat[:n_batch, :])
    var_pooled = np.dot(
        ((scores - np.dot(design, B_hat).T) ** 2),
        np.ones((int(n_array), 1)) / int(n_array),
    )
    stand_mean = np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array)))
    )

    s_data = (scores - stand_mean) / np.dot(
        np.sqrt(var_pooled), np.ones((1, int(n_array)))
    )

    batch_design = design[design.columns[:n_batch]]
    gamma_hat = np.dot(
        np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T),
        s_data.T,
    )
    delta_hat = np.array([s_data[bi].var(axis=1) for bi in batch_info])

    # eb=False: use raw gamma_hat / delta_hat
    bayesdata = s_data.copy()
    for j, batch_idxs in enumerate(batch_info):
        dsq   = np.sqrt(delta_hat[j, :]).reshape((-1, 1))
        denom = np.dot(dsq, np.ones((1, n_batches[j])))
        numer = np.array(
            bayesdata[batch_idxs]
            - np.dot(batch_design.loc[batch_idxs], gamma_hat).T
        )
        bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean

    params = dict(
        batch_levels = batch_levels,
        n_batch      = n_batch,
        grand_mean   = grand_mean,
        var_pooled   = var_pooled,
        gamma_hat    = gamma_hat,
        delta_hat    = delta_hat,
    )
    return bayesdata, params


# ---------------------------------------------------------------------------
# _combat_scores_transform: apply frozen score ComBat params to new scores
# ---------------------------------------------------------------------------

def _combat_scores_transform(scores: pd.DataFrame, batch, params: dict) -> pd.DataFrame:
    """Apply frozen eb=False ComBat parameters to new PC scores."""
    batch_levels = params["batch_levels"]
    grand_mean   = params["grand_mean"]
    var_pooled   = params["var_pooled"]
    gamma_hat    = params["gamma_hat"]
    delta_hat    = params["delta_hat"]
    n_batch      = params["n_batch"]
    n_array      = float(scores.shape[1])

    stand_mean = np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array)))
    )
    s_data = (scores - stand_mean) / np.dot(
        np.sqrt(var_pooled), np.ones((1, int(n_array)))
    )

    model_new = pd.DataFrame({"batch": batch})
    design_new = patsy.dmatrix(
        "~ 0 + C(batch, levels=%s)" % str(batch_levels),
        model_new, return_type="dataframe",
    )
    batch_design_new = design_new[design_new.columns[:n_batch]]
    batch_info_new   = model_new.groupby("batch").groups

    bayesdata = s_data.copy()
    for j, blevel in enumerate(batch_levels):
        batch_idxs = batch_info_new.get(blevel)
        if batch_idxs is None or len(batch_idxs) == 0:
            continue
        dsq   = np.sqrt(delta_hat[j, :]).reshape((-1, 1))
        denom = np.dot(dsq, np.ones((1, len(batch_idxs))))
        numer = np.array(
            bayesdata[batch_idxs]
            - np.dot(batch_design_new.loc[batch_idxs], gamma_hat).T
        )
        bayesdata[batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array)))) + stand_mean
    return bayesdata


# ---------------------------------------------------------------------------
# CovBatHarmonizer: the public sklearn-style class
# ---------------------------------------------------------------------------

class CovBatHarmonizer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible fit/transform wrapper for CovBat harmonization.

    Parameters
    ----------
    pct_var : float, default 0.95
        Cumulative variance threshold for selecting number of PCs.
    n_pc : int, default 0
        If > 0, use exactly this many PCs (overrides pct_var).
    numerical_covariates : list or None
        Indices or names of numerical columns in the model DataFrame.

    Attributes set after fit()
    --------------------------
    params_ : dict   — all fitted parameters
    npc_    : int    — number of PCs used
    is_fitted_ : bool
    """

    def __init__(
        self,
        pct_var: float = 0.95,
        n_pc: int = 0,
        numerical_covariates=None,
    ):
        self.pct_var = pct_var
        self.n_pc = n_pc
        self.numerical_covariates = numerical_covariates

        self.params_: Optional[dict] = None
        self.npc_: Optional[int] = None
        self.final_pct_var_: Optional[float] = None
        self.feat_mask_: Optional[np.ndarray] = None
        self.is_fitted_: bool = False

    # ── fit ─────────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame, batch, model=None) -> "CovBatHarmonizer":
        """
        Estimate all harmonization parameters from training data.

        Parameters
        ----------
        data  : pd.DataFrame, shape (n_features, n_samples)
        batch : array-like of batch/site labels, length n_samples
        model : pd.DataFrame or None, shape (n_samples, n_covariates)
                Biological covariates to preserve. Do NOT include batch.

        Returns
        -------
        self
        """
        _, params = _covbat_fit(
            data, batch, model=model,
            numerical_covariates=self.numerical_covariates,
            pct_var=self.pct_var, n_pc=self.n_pc,
        )
        self.params_    = params
        self.npc_       = params["npc"]
        self.final_pct_var_ = params["final_pct_var"]
        self.feat_mask_ = params["feat_mask"]
        self.is_fitted_ = True
        return self

    # ── fit_transform ───────────────────────────────────────────────────────

    def fit_transform(self, data: pd.DataFrame, batch, model=None, **fit_params) -> pd.DataFrame:
        """
        Fit on training data and return the harmonized training DataFrame.
        Output is identical to calling the original covbat() function directly.

        Parameters / Returns: same as fit().
        """
        corrected, params = _covbat_fit(
            data, batch, model=model,
            numerical_covariates=self.numerical_covariates,
            pct_var=self.pct_var, n_pc=self.n_pc,
        )
        self.params_    = params
        self.npc_       = params["npc"]
        self.final_pct_var_ = params["final_pct_var"]
        self.feat_mask_ = params["feat_mask"]
        self.is_fitted_ = True
        return corrected

    # ── transform ───────────────────────────────────────────────────────────

    def transform(self, data: pd.DataFrame, batch, model=None) -> pd.DataFrame:
        """
        Apply frozen harmonization parameters to new (test) data.
        Nothing in `data` is used to re-estimate parameters.

        Parameters
        ----------
        data  : pd.DataFrame, shape (n_features, n_samples)
                Must have the same feature rows as training data.
        batch : array-like, length n_samples
                All labels must have appeared during fit().
        model : pd.DataFrame or None — same covariate columns as used in fit().

        Returns
        -------
        corrected : pd.DataFrame, shape (n_features, n_samples)
        """
        if not self.is_fitted_:
            raise RuntimeError("Call fit() or fit_transform() before transform().")

        p       = self.params_
        n_array = float(data.shape[1])
        feat_mask = p["feat_mask"]
        
        # mask data to remove constant features
        if not feat_mask.all():
            data_orig_shape=data.shape
            data_removed=data[~feat_mask]
            data = data[feat_mask]
        
        # ── Step 1: build design matrix for new samples ──────────────────────
        if model is not None and isinstance(model, pd.DataFrame):
            mod_new = model.copy()
            mod_new["batch"] = list(batch)
        else:
            mod_new = pd.DataFrame({"batch": batch})

        mod_new = mod_new[[c for c in mod_new.columns if c not in p["drop_cols"]]]
        design_new = design_mat(mod_new, p["numerical_covariates"], p["batch_levels"])

        # ── Step 2: reconstruct stand_mean for new samples ───────────────────
        grand_mean = p["grand_mean"]
        B_hat      = p["B_hat"]
        var_pooled = p["var_pooled"]

        stand_mean_new = np.dot(
            grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, int(n_array)))
        )
        tmp = np.array(design_new.copy())
        tmp[:, : p["n_batch"]] = 0
        stand_mean_new += np.dot(tmp, B_hat).T

        # ── Step 3: standardize using training var_pooled ────────────────────
        s_data = (data - stand_mean_new) / np.dot(
            np.sqrt(var_pooled), np.ones((1, int(n_array)))
        )

        # ── Step 4: apply frozen ComBat gamma_star / delta_star ─────────────
        batch_design_new = design_new[design_new.columns[: p["n_batch"]]]
        batch_info_new   = mod_new.groupby("batch").groups
        gamma_star       = p["gamma_star"]
        delta_star       = p["delta_star"]

        bayesdata = s_data.copy()
        for j, blevel in enumerate(p["batch_levels"]):
            batch_idxs = batch_info_new.get(blevel)
            if batch_idxs is None or len(batch_idxs) == 0:
                warnings.warn(
                    f"Batch '{blevel}' was seen during fit but is absent from "
                    "this transform call.",
                    UserWarning,
                )
                continue
            dsq   = np.sqrt(delta_star[j, :]).reshape((-1, 1))
            denom = np.dot(dsq, np.ones((1, len(batch_idxs))))
            numer = np.array(
                bayesdata[batch_idxs]
                - np.dot(batch_design_new.loc[batch_idxs], gamma_star).T
            )
            bayesdata[batch_idxs] = numer / denom

        vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
        # stand_mean NOT added back yet (mirrors covbat.py internals)
        bayesdata = bayesdata * np.dot(vpsq, np.ones((1, int(n_array))))

        # ── Step 5: project onto training PCs ────────────────────────────────
        scaler = p["scaler"]
        pca    = p["pca"]
        npc    = p["npc"]

        comdata_scaled  = scaler.transform(bayesdata.T.values)    # (n_samples, n_features)
        full_scores_arr = pca.transform(comdata_scaled)            # (n_samples, n_pcs)
        full_scores     = pd.DataFrame(full_scores_arr.T)          # (n_pcs, n_samples)
        full_scores.columns = data.columns

        # ── Step 6: apply frozen score ComBat parameters ─────────────────────
        scores           = full_scores.loc[range(npc), :]
        scores_corrected = _combat_scores_transform(scores, batch, p["score_params"])
        full_scores.loc[range(npc), :] = scores_corrected

        # ── Step 7: reconstruct data ─────────────────────────────────────────
        x_covbat = bayesdata - bayesdata   # zeros, preserving index/columns
        proj = np.dot(full_scores.T.values, pca.components_)      # (n_samples, n_features)
        x_covbat += scaler.inverse_transform(proj).T
        x_covbat += stand_mean_new

        if not feat_mask.all():
            #if we masked out constant features, copy those features back into the final output
            
            x_covbat_full=pd.DataFrame(
                np.zeros(data_orig_shape),
                columns=data.columns
            )
            x_covbat_full.loc[feat_mask]=x_covbat
            x_covbat_full.loc[~feat_mask]=data_removed
            x_covbat=x_covbat_full
        
        return x_covbat

    def __repr__(self) -> str:
        status = f", npc_={self.npc_}, final_pct_var_={self.final_pct_var_}" if self.is_fitted_ else " [not fitted]"
        return f"CovBatHarmonizer(pct_var={self.pct_var}, n_pc={self.n_pc}{status})"
