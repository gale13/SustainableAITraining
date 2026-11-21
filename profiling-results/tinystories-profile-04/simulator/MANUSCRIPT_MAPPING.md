# Manuscript-to-code mapping

Source identity and line ranges are recorded in provenance.json. The private
manuscript itself is not included in this simulator package.

- Lines 310–319: J is the within-task running minimum training loss divided by
  the measured initial training loss plus epsilon.
- Lines 322–333: relative change (J_next-J)/(J+epsilon), moving-average phase
  thresholds, and nondecreasing phase with initial phase 1.
- Lines 412–415: the per-epoch error target is T*J_next.
- Lines 461–474: m_X denotes a conditional one-epoch outcome mean.

The manuscript does not uniquely prescribe an online epoch mean as its loss
instrument. The public simulator uses completed online train_loss for epoch
updates and the saved pretraining probe only for its initial anchor. See README
for the measurement limitation and for comparison with the previous version.
