#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import torchmetrics

from torchkit.core.factory import METRICS


# MARK: - Register (torchmetrics)

# MARK: Audio Metrics
METRICS.register(name="pit",
                 module=torchmetrics.PermutationInvariantTraining)
METRICS.register(name="permutation_invariant_training",
                 module=torchmetrics.PermutationInvariantTraining)
METRICS.register(name="si_sdr",
                 module=torchmetrics.ScaleInvariantSignalDistortionRatio)
METRICS.register(name="scale_invariant_signal_distortion_ratio",
                 module=torchmetrics.ScaleInvariantSignalDistortionRatio)
METRICS.register(name="si_snr",
                 module=torchmetrics.ScaleInvariantSignalNoiseRatio)
METRICS.register(name="scale_invariant_signal_noise_ratio",
                 module=torchmetrics.ScaleInvariantSignalNoiseRatio)
METRICS.register(name="sdr",
                 module=torchmetrics.SignalNoiseRatio)
METRICS.register(name="signal_distortion_ratio",
                 module=torchmetrics.SignalDistortionRatio)
METRICS.register(name="snr",
                 module=torchmetrics.SignalNoiseRatio)
METRICS.register(name="signal_noise_ratio",
                 module=torchmetrics.SignalNoiseRatio)

# MARK: Classification Metrics
METRICS.register(name="accuracy",           module=torchmetrics.Accuracy)
METRICS.register(name="average_precision",  module=torchmetrics.AveragePrecision)
METRICS.register(name="auc",                module=torchmetrics.AUC)
METRICS.register(name="auroc",              module=torchmetrics.AUROC)
METRICS.register(name="binned_average_precision",
                 module=torchmetrics.BinnedAveragePrecision)
METRICS.register(name="binned_precision_recall_curve",
                 module=torchmetrics.BinnedPrecisionRecallCurve)
METRICS.register(name="binned_recall_at_fixed_precision",
                 module=torchmetrics.BinnedRecallAtFixedPrecision)
METRICS.register(name="calibration_error",
                 module=torchmetrics.CalibrationError)
METRICS.register(name="cohen_kappa",        module=torchmetrics.CohenKappa)
METRICS.register(name="confusion_matrix",   module=torchmetrics.ConfusionMatrix)
METRICS.register(name="f1",                 module=torchmetrics.F1Score)
METRICS.register(name="f1_score",           module=torchmetrics.F1Score)
METRICS.register(name="fbeta",              module=torchmetrics.FBetaScore)
METRICS.register(name="fbeta_Score",        module=torchmetrics.FBetaScore)
METRICS.register(name="hamming_distance",   module=torchmetrics.HammingDistance)
METRICS.register(name="hinge",              module=torchmetrics.Hinge)
METRICS.register(name="hinge_loss",         module=torchmetrics.HingeLoss)
METRICS.register(name="iou",                module=torchmetrics.IoU)
METRICS.register(name="jaccard_index",      module=torchmetrics.JaccardIndex)
METRICS.register(name="kl_divergence",      module=torchmetrics.KLDivergence)
METRICS.register(name="matthews_corr_coef", module=torchmetrics.MatthewsCorrCoef)
METRICS.register(name="precision",          module=torchmetrics.Precision)
METRICS.register(name="precision_recall_curve",
                 module=torchmetrics.PrecisionRecallCurve)
METRICS.register(name="recall",             module=torchmetrics.Recall)
METRICS.register(name="roc",                module=torchmetrics.ROC)
METRICS.register(name="specificity",        module=torchmetrics.Specificity)
METRICS.register(name="stat_scores",        module=torchmetrics.StatScores)

# MARK: Image Metrics
METRICS.register(name="psnr",
                 module=torchmetrics.PeakSignalNoiseRatio)
METRICS.register(name="peak_signal_noise_ratio",
                 module=torchmetrics.PeakSignalNoiseRatio)
METRICS.register(name="ssim",
                 module=torchmetrics.StructuralSimilarityIndexMeasure)
METRICS.register(name="structural_similarity_iIndex_measure",
                 module=torchmetrics.StructuralSimilarityIndexMeasure)
METRICS.register(name="multi_scale_ssim",
                 module=torchmetrics.MultiScaleStructuralSimilarityIndexMeasure)
METRICS.register(name="multi_scale_structural_similarity_index_measure",
                 module=torchmetrics.MultiScaleStructuralSimilarityIndexMeasure)

# MARK: Detection Metrics
#METRICS.register(name="bbox_map", module=torchmetrics.MAP)

# MARK: Regression Metrics
METRICS.register(name="cosine_similarity",
                 module=torchmetrics.CosineSimilarity)
METRICS.register(name="explained_variance",
                 module=torchmetrics.ExplainedVariance)
METRICS.register(name="mean_absolute_error",
                 module=torchmetrics.MeanAbsoluteError)
METRICS.register(name="mean_absolute_percentage_error",
                 module=torchmetrics.MeanAbsolutePercentageError)
METRICS.register(name="mean_squared_error",
                 module=torchmetrics.MeanSquaredError)
METRICS.register(name="mean_squared_log_error",
                 module=torchmetrics.MeanSquaredLogError)
METRICS.register(name="pearson_corr_coef",
                 module=torchmetrics.PearsonCorrCoef)
METRICS.register(name="r2_score",
                 module=torchmetrics.R2Score)
METRICS.register(name="spearman_corr_coef",
                 module=torchmetrics.SpearmanCorrCoef)
METRICS.register(name="symmetric_mean_absolute_percentage_error",
                 module=torchmetrics.SymmetricMeanAbsolutePercentageError)
METRICS.register(name="tweedie_deviance_score",
                 module=torchmetrics.TweedieDevianceScore)

# MARK: Retrieval Metrics
METRICS.register(name="retrieval_fallout",
                 module=torchmetrics.RetrievalFallOut)
METRICS.register(name="retrieval_hit_rate",
                 module=torchmetrics.RetrievalHitRate)
METRICS.register(name="retrieval_map",
                 module=torchmetrics.RetrievalMAP)
METRICS.register(name="retrieval_mrr",
                 module=torchmetrics.RetrievalMRR)
METRICS.register(name="retrieval_normalized_dcg",
                 module=torchmetrics.RetrievalNormalizedDCG)
METRICS.register(name="retrieval_precision",
                 module=torchmetrics.RetrievalPrecision)
METRICS.register(name="retrieval_recall",
                 module=torchmetrics.RetrievalRecall)
METRICS.register(name="retrieval_r_precision",
                 module=torchmetrics.RetrievalRPrecision)

# MARK: Text Metrics
METRICS.register(name="bleu_score",       module=torchmetrics.BLEUScore)
METRICS.register(name="char_error_rate",  module=torchmetrics.CharErrorRate)
METRICS.register(name="chrf_score",       module=torchmetrics.CHRFScore)
METRICS.register(name="extended_edit_distance",
                 module=torchmetrics.ExtendedEditDistance)
METRICS.register(name="match_error_rate", module=torchmetrics.MatchErrorRate)
METRICS.register(name="sacre_blue_score", module=torchmetrics.SacreBLEUScore)
METRICS.register(name="squad",            module=torchmetrics.SQuAD)
METRICS.register(name="translation_edit_rate",
                 module=torchmetrics.TranslationEditRate)
METRICS.register(name="wer",              module=torchmetrics.WordErrorRate)
METRICS.register(name="word_error_rate",  module=torchmetrics.WordErrorRate)
METRICS.register(name="wil",              module=torchmetrics.WordInfoLost)
METRICS.register(name="word_info_lost",   module=torchmetrics.WordInfoLost)
METRICS.register(name="wip",
                 module=torchmetrics.WordInfoPreserved)
METRICS.register(name="word_info_preserved",
                 module=torchmetrics.WordInfoPreserved)
