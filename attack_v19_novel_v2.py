#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Coherent Spatial-Lie Attack on VLMs — WHITE-BOX (v18.11 + DICD + Fast Batched)

 DICD  DUAL-INVERSION COUNTERFACTUAL DECOMPOSITION  (All 5 Spatial Families)
 ---------------------------------------------------------------------------
 Generalises MBPD + CCT + PGCM from lateral-only to ALL five VSR spatial
 families (lateral, vertical, depth, containment, proximity) without external
 detectors, bounding-box cut-and-paste seams, or upside-down gravity violations:
   (1) PROPOSITION-DUAL INVERSION (T_prop, Yes/No channel, all 5 families):
       Within any yes/no frame f on the unedited clean image x_clean, querying
       the true relation q_true = f(s, o, r_true) has visual truth +P_f, while
       querying the target relation q_target = f(s, o, r_target) has visual
       truth -P_f under the identical frame assent bias A_f:
           A_f^dual = (g_clean(f, r_true) + g_clean(f, r_target)) / 2
           P_f^dual = (g_clean(f, r_true) - g_clean(f, r_target)) / 2
       Validated across 140 lateral (example, frame) pairs: P_f^dual correlates
       with P_f^mirror at Pearson r = +0.915 (true-Q) and +0.917 (target-Q),
       and A_f^dual correlates with A_f^mirror at r = +0.873.
   (2) CONVERSE ARGUMENT-SWAP + POLARITY-DUAL PGCM (Open channel, all families):
       For anti-symmetric families (vertical, depth), s R_true o <=> o R_target s,
       so probing "The <obj> is ___" on x_clean under the converse open question
       gives the target-direction teacher margin m0 and percept trust beta; if
       commonsense noun asymmetry suppresses the converse slot (or on symmetric /
       topological families proximity & containment), PGCM falls back to the
       clean open polarity-dual margin (-clean_dir_gap).
   (3) COHERENCE & ON-POLICY REFINEMENTS (from the 20-ex all-family runs):
       * Relation-Preserving On-Policy Minting (split_prefix_suffix): fixes the
         fixed k=4 slice that chopped the target relation word ('far', 'left',
         'right', 'behind', 'on') off tgt_ids[4:] whenever <subj> was 1 token,
         and strips contradicted relation words from the wrong prefix.
       * Symmetric Shortfall-Aware Fragility (MBPD_SHORTFALL_WEIGHT): boosts
         both 'No' rows fighting assent bias (n6 'Can you confirm...') and
         'Yes' rows fighting dissent bias (y2/y9/y11 'Would it be accurate...').
       * Bias-Cancelling Teacher + Label-Safe VCD Head: keeps CCT active with
         capped alpha on perception_dead rows with strong frame bias and ensures
         the VCD plausibility mask never zeroes out the target label token.
       * Token-0 Yes/No Pivot Mask + Multi-Word Preposition Pivot ('in front',
         'on top'): marks token 0 in 'Yes.'/'No.' and leading prepositions in
         multi-word target relations so CW and contrast margins act at the true
         branch point right after "The <subj> is".
   (4) LOSSLESS 5x-7x SPEEDUP (Zero Accuracy Compromise):
       * Vision-Tower Batch Deduplication + no_grad Feature Cache (runs ViT on
         batch_size=1 and expands to B instead of running ViT B times).
       * Fast Tokenization Cache (skips ~350 redundant CPU CLIPImageProcessor
         calls per image).
       * Auto-scaled MAX_ROWS_PER_FWD=28 + batched _preserve_loss + forward-only
         (no_grad) f_best refreshes + batched yesno_gap_batch & generate_batch.

 v18.11  MBPD LOSS  (Survivor 1 promoted from diagnostic to objective)
 ---------------------------------------------------------------------
 Evidence (v18.10 pilot, 3 lateral examples, 84 frame x question cells):
   * pre-attack perception P = (g_mirror - g_clean)/2 predicts the FINAL
     adversarial margin of the true-question ("No") rows: Spearman 0.77
     (bed 0.68, laptop 0.76, dog 0.83); the clean gap alone does NOT (-0.37)
     and the assent bias alone does not (0.12);
   * mean final margin on true-Q rows: mirror_enough 4.12 > beyond_mirror
     2.14 > perception_dead 0.74 (min 0.06); the held-out assent frame n6 is
     among the weakest cells in every example.
 So the weak points of the lie are known BEFORE the attack. v18.11 uses that:
   (1) PER-FRAME TEACHER STRENGTH. Before the attack, every TRAIN yes/no frame
       is classified from 2 no-grad gaps (clean, mirror) exactly as in the
       v18.10 diagnostic. The CCT target then uses, per row,
         prior_already_lie -> no teacher (the clean image already says the lie)
         mirror_enough     -> alpha = 0   (plain mirror target)
         beyond_mirror     -> alpha = alpha_f from the closed form
                              1+alpha_f = (m - s*g_clean)/(s*(g_mirror-g_clean)),
                              capped at MBPD_ALPHA_CAP
         perception_dead   -> no teacher (there is nothing perceptual to copy)
       instead of one global CF_CONTRAST_ALPHA for every row.
   (2) FRAGILITY WEIGHTING. "No" probes are up-weighted by
         w *= 1 + MBPD_FRAG_W * sigmoid((MBPD_FRAG_P0 - P_f) / MBPD_FRAG_S)
       so low-perception frames (which ended with the thinnest margins) get
       more of the budget. High-P frames keep weight ~1.
 Survivor 2 ("carried_by perception/bias") is DROPPED as a claim: it was
 inconsistent across the 3 pilot examples (28/0, 1/27, 13/15). The raw dA/dP
 numbers are still logged as unvalidated telemetry.
 ABLATION (same seed, CRN on): A2 = v18.10 (MBPD_LOSS=False) vs B = v18.11.
 Success = n6/n11-type "No" rows finish with larger margins (read mbpd_diag)
 and train-type examples flip their no-probes, at similar preserve_f1.

 v18.10  MBPD DIAGNOSTIC (Mirror Bias-Perception Decomposition) -- measurement
 only, NOTHING enters the gradient. For every yes/no frame f (all 14) and both
 questions q (target relation, true relation) it measures the first-token gap
 g = log p(Yes) - log p(No) on the clean image, the mirrored clean image, the
 final adversarial image and the mirrored adversarial image, then splits it:
     A_f,q = (g_mirror + g_clean) / 2    assent / prior bias  (perception cancels)
     P_f,q = (g_mirror - g_clean) / 2    perception strength  (bias cancels)
 Mirroring flips what the model SEES but not how much it tends to AGREE, so the
 sum isolates bias and the difference isolates perception -- no blank image
 and no extra model needed. With s = +1 (target-Q must say Yes) or -1 (true-Q
 must say No), each (f,q) is classified:
     mirror_enough    s*g_mirror >= MBPD_MARGIN     the mirror already states the lie
     beyond_mirror    perception exists but assent bias beats it; the closed-form
                      extrapolation that would reach the margin is
                      1+alpha_f = (MBPD_MARGIN - s*g_clean) / (s*(g_mirror-g_clean))
     perception_dead  s*(g_mirror-g_clean) < 2*MBPD_P_MIN : the model barely
                      sees the flip in this frame; no teacher can help here
 (plus prior_already_lie: s*g_clean >= MBPD_MARGIN, the clean image already
 says the lie through bias alone -- typical for target-Q under yes-bias)
 and the adversarial outcome adv_ok = (s*g_adv > 0) is recorded next to it, so
 the hypothesis "frames where assent beats perception are the ones that fail"
 can be checked directly. The (adv, flip(adv)) pair is decomposed the same way
 (dA, dP): a lie carried by perception moves P, a yes/no bias hack moves A.
 Cost: 4 images x 14 frames x 2 questions = 112 no-grad forwards per example.
 Output: results.json -> "mbpd_diag"; summarise with mbpd_report.py.

 v18.9  CONTRASTIVE COUNTERFACTUAL TEACHER (CCT)  -- new teacher loss
 ----------------------------------------------------------------------
 v18.8 distilled the MIRROR's full next-token distribution (plain KL, the same
 mechanism as Prompt Matching / Hinton KD). Two measured problems:
   (i)  mean KL was only 0.08-0.21: most answer positions ("Yes", ",", "it
        would be accurate to say") are identical on mirror and clean, so the
        loss is dominated by tokens that carry no relation information;
   (ii) the hard yes/no gate failed on yes-biased scenes (mirror says Yes/Yes)
        and those examples lost the teacher entirely (bed, train).
 CCT replaces the teacher target by the COUNTERFACTUAL CONTRAST
        z* = (1+a) * z_mirror - a * z_clean          (on the mirror's plausible
                                                      head, VCD-style constraint)
 and weights every answer position by the Jensen-Shannon divergence between
 the mirror and clean distributions at that position:
        w_i = clip( JS(p_mirror,i || p_clean,i) / CF_CONTRAST_JS_REF, 0, 1 )
        L   = T^2 * (1/alen) * sum_i  w_i * KL( softmax(z*_i / T) || student_i )
 * Positions where mirror == clean get w ~ 0  -> only relation-bearing tokens
   are distilled.  (fixes i)
 * A scene the model does not perceive (mirror and clean BOTH say "Yes") gives
   JS ~ 0 everywhere, so the term silences itself: the contrast IS the
   perception gate. The hard gate is therefore no longer required for the
   yes/no teacher (CF_CONTRAST_REPLACE_GATE).  (fixes ii)
 * The contrast cancels shared biases (yes-bias, language prior) that the plain
   mirror teacher would otherwise copy into the attacked model.
 Contrastive decoding (Li et al. 2023) and VCD (Leng et al. 2024) use this
 contrast at DECODING time as a hallucination defence; here it is the
 OPTIMISATION TARGET of an input perturbation, with a label-known symmetry
 (horizontal mirror) as the counterfactual.
 Open rows (CF_OPEN_DIR + PGCM): the percept trust beta additionally uses the
 CONTRASTIVE swing  dir_gap(mirror) - dir_gap(clean)  instead of the mirror's
 gap alone, so a mirror that is only "biased toward the target word" without
 an actual perceptual swing is down-weighted.

 ABLATION (same seed, CRN on, same examples):
   A0  LOSS_MODE="classic"
   A1  LOSS_MODE="cf_hybrid", CF_CONTRAST=False      (v18.8 mirror teacher)
   A2  LOSS_MODE="cf_hybrid", CF_CONTRAST=True       (v18.9 CCT)
   A3  = A2 with CF_CONTRAST_ALPHA in {0.5, 2.0}; A4 = A2 with JS weight off
 Read cf_stats.mean_contrast_w and cf_info.contrast_* per example.
================================================================================
 (original v18 notes follow)
   = v17 (CW + contrast + evasion + hard-probe + free-run)
       + IDEA-2  ON-POLICY PREFIX AUGMENTATION   (scheduled-sampling for attacks)
       + IDEA-3  PARAPHRASE POOL + WORST-CASE DRO (CVaR) + leak-free eval split

WHAT v18 ADDS AND WHY
---------------------
Diagnosis of the v17 run (23 examples, 161 probe results, 24 misses):
    Does X appear <rel_true>? -> "No"     14/23 = 0.609   9 misses  (all say "Yes")
    In one short phrase, where is X ...?  14/23 = 0.609   9 misses  (all "next to" dodges)
    axis / relative / is-it-correct       0.913-1.000     6 misses
Both failing buckets were HELD OUT of the training bank; 4 of the 7 eval probes
were verbatim members of it. Split by that:  train-seen 0.935 vs held-out 0.739.
The reported 0.851 averaged the two.

IDEA-2  ON-POLICY PREFIX AUGMENTATION  (targets the "next to" dodges)
    OPT-C only ever scored ONE teacher-forced token. A dodge is a whole clause.
    Every ONPOLICY_EVERY steps we greedily decode the CURRENT x_adv, and for any
    open probe that fails score_lie() we mint a new training row whose CONTEXT is
    the model's own wrong prefix and whose LABELS are the target continuation:
        ctx = "USER: <image> {q} ASSISTANT: The bowl is next"
        lbl = "to the right of the banana."
    The perturbation is then optimised to RECOVER onto the lie from the model's
    real off-target trajectory. This is scheduled sampling / DAgger applied to
    adversarial optimisation. Rows live in a FIFO buffer so we track a moving
    failure mode instead of overfitting one prefix.
    Failing yes/no probes cannot be prefix-augmented (target "No." is 2 tokens),
    so instead their OPT-B difficulty EMA is boosted, which makes CVaR below pick
    them up on the next step.

IDEA-3  PARAPHRASE POOL + CVaR + DISJOINT EVAL  (targets the appear-frame gap)
    (a) 20 open templates and 14 yes/no frames replace the 3 hand-written
        variants. A FIXED seed splits both pools into disjoint TRAIN and EVAL
        halves, so no eval probe is ever optimised. Reported SBC becomes a real
        held-out generalisation number.
    (b) PROMPT_EOT is replaced by a stratified subset draw of DRO_SUBSET probes
        per step (cost held at the v17 level), and uniform averaging is replaced
        by CVaR: only the worst ceil(alpha*n) probes by difficulty EMA receive
        gradient, with alpha annealed 1.0 -> CVAR_ALPHA_FINAL. Coherence is a MIN
        over probes, so a mean objective has no reason to lift the weakest one;
        this does. Zero-weight probes are skipped entirely, so CVaR is cheaper
        than uniform, not more expensive.

SCORER (v18.1) -- fixes measured on the first v18 run (n=5)
    The v18 held-out templates elicit longer answers than the v17 probes did,
    and the substring scorer broke in BOTH directions on them:
      false NEGATIVE  "The cow is walking in front of the sheep, with the sheep
                       following behind it."  -> scored a MISS because the true
                       relation "behind" appears in a subordinate clause about
                       the OBJECT, not the subject. The target relation is stated
                       correctly; this is a hit.
      false POSITIVE  "The bowl is not to the right of the banana." -> scored a
                       HIT because "right" is present. Negation was ignored.
      degenerate HIT  "No. The image does not show a bed."  -> scored a HIT on a
                       "no" probe. The model is not holding a false belief about
                       WHERE the bed is; it no longer sees a bed. Counting this
                       as success inflates SBC exactly where preservation has
                       collapsed, which is the one thing this paper must not do.
    score_open() now (a) splits the answer into clauses, (b) scopes the check to
    the clause that mentions the SUBJECT, (c) rejects a target match that is
    negated, and (d) flags degenerate answers so they can be reported separately.
    SBC_nondegen is the honest headline; SBC is kept for comparability.

Both are flag-gated (ONPOLICY, PARAPHRASE_POOL, DRO_CVAR). Turn all three off and
you get bit-identical v17 behaviour, so every ablation stays runnable.
SCORER_STRICT=False restores the v17/v18.0 substring scorer for ablation.

v18.2 -- driven by the v18.1 failure analysis
---------------------------------------------
The v18.1 run (n=5) showed 5 of 7 misses were scorer reason `target_absent`, and
the answers explain why:
    "The cow is located to the left of the man holding the baseball bat."
    "The potted plant is located to the right of the man's head."   (asked: bed)
There is no man and no baseball bat. The model is not dodging the question, it
is describing a DIFFERENT IMAGE. At eps=16/255 the perturbation REPLACES the
scene rather than editing one relation. That single fact explains both remaining
failure modes at once -- the open-probe misses AND preserve_strict=0.000 -- and
it means no further loss term can help: you cannot push a model toward a
relation between two objects it can no longer see.

So v18.2 adds three things that act on the perturbation itself, not the loss:

  MINT GATE (fixes a real bug in IDEA-2)
    _onpolicy_update() minted a row whenever score_lie() failed, without asking
    WHY it failed. On bed/potted-plant that meant 46 rows minted from
    hallucinated scenes -- teaching the perturbation to "recover onto the lie"
    from a prefix that describes an image that does not exist. Wasted budget on
    exactly the hardest example. Now a roll-out is only minted if it is a real
    dodge: non-degenerate, and the subject is actually named.

  SPATIAL MASK (new)
    Freeze most of the image. A saliency mask from the loss gradient at x_clean
    keeps the top MASK_KEEP_FRAC of pixels editable (blurred and dilated into
    coherent regions) and pins the rest to the clean image exactly. The scene
    cannot be replaced if 85% of the pixels cannot move, which gives
    preserve_strict its first actual path off zero.

  CURVE MODE (new)
    CURVE_MODE="eps" crafts each example at every eps in EPS_CURVE_VALUES and
    records ALL of them, not the best. Same for CURVE_MODE="mask" over
    MASK_CURVE_VALUES. This produces the coherence-vs-fidelity Pareto curve,
    which is the headline figure for the reframed paper. It is NOT a best-of
    sweep -- selecting the best eps per example would be selection on the test
    metric, which is the thing we are trying to stop doing.

EXPECTATION, stated in advance so it cannot be rationalised afterwards: masking
and lower eps REDUCE the attack's capacity. SBC is expected to FALL and
preservation to RISE. The claim being tested is not "this raises SBC" -- it is
"there exists a budget at which the lie is coherent AND the scene survives". If
no such budget exists, that is a publishable negative result about targeted VLM
belief editing, and it is worth more than another 0.80.

v18.3 -- objective-side fixes (this file)
-----------------------------------------
Diagnosis of the v18.2 run (n=5, results__53_): the ONLY live v18.2 change was
the mint gate, which rejected 3 of 126 open failures (2.4%) -- and 0 of the 46
rows on bed/potted-plant, the example it was written for. Reading the v18.1
rollouts for that example, 45 of the 46 minted rows came from GENUINE dodges
("The bed is located next to the potted plant."), not hallucinated scenes. The
v18.2 header conflated "46 rows minted on the hardest example" with "46 rows
minted from hallucinated scenes". The 46 was never evidence for the hypothesis.

So v18.3 stops patching the constraint set and fixes the OBJECTIVE. The guiding
table is the gap between what is optimised and what is measured:

    decoding      teacher-forced logits      vs   free-running greedy
    prompts       ~10 train templates,            7 HELD-OUT templates
                  CVaR-narrowed to ~3
    aggregation   weighted sum (soft)        vs   AND over probes (coherence)
    preservation  NOTHING                    vs   preserve_f1 / preserve_strict

Row 4 is free to close and v18.2 never closed it. Row 2 CVaR actively makes
worse. Rows 1 and 3 were already handled (OPT-C, IDEA-2, soft-min).

  (A) PRESERVATION LOSS + LAGRANGIAN  [PRESERVE_LOSS]
      preserve_f1 sat at 0.60-0.68 and preserve_strict at 0.000 across every
      run since v15, and NOTHING in _row_loss ever touched either. The spatial
      mask was a constraint-side patch for an objective-side omission, and
      constraint-side patches cost SBC by construction.
      Fix: before the attack, greedily decode the CLEAN model's answers to the
      PRESERVATION_QUESTIONS and freeze them as labels. Add
          L_preserve = mean_q CE(logits_adv(q), clean_greedy_answer(q))
      to the objective. We are then optimising exactly the quantity we report.
      With PRESERVE_LAGRANGIAN the weight is not a hand-tuned constant but a
      dual variable solving
          min_delta  L_lie   s.t.  L_preserve <= PRESERVE_C
      via  lambda <- clip(lambda + LR * (L_preserve - C), 0, LAMBDA_MAX).
      Sweeping PRESERVE_C then traces the coherence-vs-fidelity Pareto curve
      DIRECTLY, one run per point. That is a stronger figure than the eps or
      mask curve, because each point is a controlled trade-off rather than a
      capacity throttle: the mask says "you may not touch 85% of pixels", the
      constraint says "touch anything, as long as the caption survives".
      Note: lambda moving makes the objective non-stationary, exactly as
      on-policy minting does. PRESERVE_REFRESH_BEST re-measures APGD's f_best
      after a material lambda change, for the same reason ONPOLICY_REFRESH_BEST
      exists. Keep PRESERVE_LAMBDA_LR small.

  (B) CE ON NON-PIVOT POSITIONS ONLY  [CE_NONPIVOT_ONLY]
      A pivot position previously received BOTH an unbounded CE gradient and a
      hinge that switches off at MARGIN. Carlini & Wagner's result is that the
      margin loss beats CE for targeted attacks precisely because CE never
      stops pushing once the argmax is decided. That surplus pressure is what
      put "a cup of coffee" and "covered in writing or drawings" into the v18.2
      adversarial captions -- the model verbalising a perturbation that kept
      growing after it had already won.
      Now: CE owns the carrier words (fluency), the hinge owns the relation
      word (sufficiency, with a hard stop). Unbounded pressure nowhere;
      sufficient pressure at exactly one place. Falls back to full-sequence CE
      whenever the pivot mask is all-True (yes/no probes) so nothing is starved.

  (C) YESNO_CONTRAST ON BY DEFAULT  [V183_YESNO_CONTRAST]
      With it False, _ids_for("no") returned (None, None), so use_contrast was
      False on EVERY yes/no probe -- the contrast term literally never ran
      there. All five no-probe misses in the v18.2 run were "Yes" beating "No"
      at token 0 under held-out confirmation frames ("Can you confirm that...",
      "Would it be accurate to say..."). This flag puts a margin-6 hinge on
      exactly z_No - z_Yes across both casings. It reuses an existing code path
      and costs nothing.

  (D) APGD cond1 FIXED  [APGD_COND1_MODE="prev"]
      The v18.2 loop counted improvement over the GLOBAL BEST. Croce & Hein
      count improvement over the PREVIOUS ITERATE:
          sum_i 1[f(x^(i+1)) > f(x^(i))] < rho * (w_j - w_{j-1})
      Improving on a running best is strictly harder, so cond1 fired too often,
      eta halved prematurely and the search froze early -- a plausible partial
      cause of the terminal losses of 149-171 on the failing examples versus
      41.6 on the one that succeeded. Set APGD_COND1_MODE="best" to reproduce
      the v18.2 behaviour exactly.
      Iterations whose objective just changed (a mint, or a material lambda
      step) are EXCLUDED from the improvement count instead of being scored
      against a loss that no longer exists.

  (E) BUDGET  [STEPS_SCALE]
      100-150 steps is low for a targeted VLM attack, and it starves IDEA-2:
      at 100 steps with ONPOLICY_EVERY=10 and ONPOLICY_START_FRAC=0.20 you get
      8 minting rounds into a buffer of capacity 10, which never turns over.
      STEPS_SCALE multiplies every family budget in one place. Left at 1.0 by
      default so this file is comparable to v18.2 out of the box; the
      recommended diagnostic is PRESET="steps_probe" on ONE example.

  (F) THE ABLATIONS THAT WERE NEVER RUN
      _ABLATIONS tested the v18 ideas only. It never tested the five loss terms
      themselves, so five hand-weighted terms carried zero ablation evidence.
      Added: ablate_noce, ablate_nomargin, ablate_nocontrast, ablate_noevasion,
      ablate_preserve, ablate_nopreserve, cvar_035 / cvar_060 / cvar_100.
      On CVaR specifically -- alpha anneals to 0.35, so the last half of the run
      puts gradient on ceil(0.35*6)=3 TRAINING prompts while the metric is 7
      DISJOINT ones. Worst-case reweighting is the right tool when eval == train
      and the wrong tool when they are disjoint by construction; the correct
      tool for a generalisation gap is augmentation. sbc_yes=1.000 with
      sbc_no=0.500, all five misses in one held-out frame, is the signature of
      fitting a narrow prompt distribution. PREDICTION, recorded in advance:
      cvar_100 (uniform) generalises better; cvar_035 wins on train-seen probes.

  STILL TRUE AND STILL THE FIRST THING TO DO: at n=5 a 3-row change moved SBC
  by 0.114. Run the SAME config over >=3 seeds and report the spread before
  attributing anything above to anything below. If the spread is +-0.10, the
  honest statement is that nothing since v15 is distinguishable.

  NOT DONE ON PURPOSE: no sixth loss term for the depth-family failures. The
  n=23 run gives depth coherence 0.167, vertical 0.000, proximity 0.000 against
  lateral 0.417, and the model answered on the LATERAL axis three separate
  times when asked for DEPTH. No logit margin fixes a missing representation.
  Report the per-family breakdown as a finding.
================================================================================
WHY v15 (diagnosis from v14's probe_log)
----------------------------------------
v14 moved coherence (0.30 -> 0.40) via hard-probe weighting, but sbc_open stayed
pinned at exactly 0.667 — one of every three open probes fails, every time — and
that ceiling caps coherence. Reading the raw failing answers:
   * 16 of 20 open failures are DODGES that reframe from the FIRST content token:
       "The bowl is NEXT to the banana", "The teddy bear is SMALLER than the bench",
       "The cake is SMALLER than the teddy bear", "positioned in a way that ..."
   * only 4 leak the true relation.
   * the forced-choice axis question almost never fails; the "describe" questions do.
v14's OPT-A suppresses these escape WORDS, yet they still appear — because every
v14 term lives on TEACHER-FORCED positions of OUR target sentence. At generation
time the model leaves that template at token ~2 ("The bowl is [next]"), a branch
the teacher-forced loss never scores. Teacher forcing hides the divergence.

THE KEY FACT v15 EXPLOITS
-------------------------
The FIRST teacher-forced position whose argmax != target is EXACTLY the token
greedy decoding would emit wrong first. Proof: while argmax == target at every
earlier position, greedy reproduces the target prefix, so the contexts stay
identical; the first disagreement is therefore the real greedy branch point.
=> We can find and attack the model's own divergence token WITH NO GENERATION,
   fully differentiably, from the teacher-forced logits we already compute.

WHAT v15 ADDS
-------------
OPT-C  Free-run divergence margin.
       Per reference, mark the first position i* where argmax != label and pour a
       strong CW margin into it:  relu(FREERUN_MARGIN - (z_target[i*] - z_wrong[i*])),
       where z_wrong is the competitor the model actually prefers there. This
       covers ALL failure modes at the one decisive spot — dodge ("next"),
       off-axis ("in front"), true-leak, and yes-to-true (a "No" that should be
       "Yes" diverges at token 0) — so it lifts sbc_open, sbc_yes/no, and hence
       coherence together. Vectorized (no host sync), so it is cheap; set
       FREERUN_WEIGHT=0 to recover exact v14 behavior.

Everything else (APGD, soft-min multi-ref, forced-choice, OPT1 pivot+contrast,
OPT-A evasion, OPT-B hard-probe, STE quant, projection, eval, checkpoint/resume,
Drive mirror, seeding, and the v12 memory path) is unchanged.

Scope: evaluates a model you load and fully control on a public benchmark (VSR).
NOTE: raise NUM_EXAMPLES to >=300 and run >=3 seeds before reporting any table.
      Report held-out SBC, coherence, preserve_f1 and preserve_strict TOGETHER.
================================================================================
"""

import os, sys, gc, glob, io, json, math, random, hashlib, subprocess, time, shutil, zlib
import re as _re

# --- memory: reduce allocator fragmentation. MUST be set before torch/CUDA init.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HF_CACHE_DIR = "/content/drive/MyDrive/hf_cache"
if os.path.isdir("/content"):
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)


def _ensure_deps(extra=None):
    pkgs = []
    for mod, spec in {"transformers": "transformers>=4.52", "accelerate": "accelerate",
                      "pandas": "pandas", "datasets": "datasets"}.items():
        try: __import__(mod)
        except Exception: pkgs.append(spec)
    for spec in (extra or []): pkgs.append(spec)
    if pkgs:
        print("[setup] installing:", pkgs)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)
    else:
        print("[setup] base dependencies present.")


_ensure_deps()

import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image

try:
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        print("[setup] mounting Google Drive ..."); drive.mount("/content/drive")
    else:
        print("[setup] Drive already mounted.")
except Exception:
    print("[setup] not in Colab — continuing.")


# =============================================================================
# 1. CONFIG (white-box; CW + contrast + evasion + hard-probe + STE/EOT)
# =============================================================================
class CFG:
    # =========================================================================
    # v17 PRESET.  Set this ONE string; it overwrites the flags at the bottom.
    #
    #   "v15"   EXACT v15 baseline. Reproduces SBC 0.836 on the 20-ex pilot.
    #           <- DEFAULT, because the v16 flags measured NET NEGATIVE (-0.190
    #              SBC on a same-seed, same-fr, 3-example controlled A/B).
    #   "v16"   all v16 changes on (kept only so the regression is reproducible)
    #   "ablate_<flag>"  v15 baseline + exactly ONE v16 change. This is the only
    #           honest way to test them; v16 changed four things at once and the
    #           result is uninterpretable. Valid:
    #              ablate_appear      (adds the 'Does X appear' training frame)
    #              ablate_yesno_ctr   (yes/no polarity contrast)
    #              ablate_yesno_refs  (multi-reference yes/no answers)
    #              ablate_evasion     (adds 'located'/'situated')
    #              ablate_restart     (restart sweep instead of fr sweep)
    #   "custom"  honour whatever is written below; no preset overwrite.
    # =========================================================================
    # v18.7 IDEA-A, ON. "lagr_crn" is in the _v183_baseline whitelist, so the
    # full v18.3/v18.6 objective is applied FIRST and this preset then changes
    # exactly the allocation rule:
    #     PROBE_LAGRANGIAN True | DRO_CVAR, HARDPROBE_WEIGHTING,
    #     FAMILY_REWEIGHT False | CRN_ENABLE, CRN_STRICT_PROBES True
    # Do NOT set PROBE_LAGRANGIAN=True by hand instead -- that leaves hardprobe
    # weighting stacked on top of the duals and measures two allocation rules.
    # The matching CONTROL arm is PRESET="famrw_crn" (same objective, same
    # streams, v18.6 allocation). Run both; one arm alone measures nothing.
    PRESET = "reclaim"
    # =========================================================================
    # STEP 3 (v19): BASELINES & FULL METHOD SUITE ON TEST SPLIT
    #   Set BASELINE_ARM (or pass `--arm <name> --seed <int>` on the CLI):
    #     "full" -> Full method (v19: on-policy minting + CCT/DICD + MBPD + preservation)
    #     "B0"   -> Clean image & random uniform L_inf noise at eps (0 steps, floor)
    #     "B1"   -> Single-prompt targeted PGD (CE on 1 open question + 1 target answer)
    #     "B2"   -> Image Hijacks behaviour matching (Bailey et al., ICML 2024:
    #               CE on target answers over the full train probe pool, no extras)
    #     "B3"   -> Image Hijacks Prompt Matching (teacher = clean image + text
    #               "Note: in this scene, the <subj> is <rel_target> the <obj>.", KL distil)
    #     "B4"   -> CroPA (Luo et al., ICLR 2024: learnable adversarial prompt
    #               perturbation + cross-prompt image PGD)
    #     "B5"   -> Feature matching to mirrored image (AttackVLM-style, Zhao et al.,
    #               NeurIPS 2023: MSE on vision-encoder patch features)
    #   Set RUN_SUITE = ["B0", "B1", "B2", "B3", "B4", "B5", "full"] to run multiple
    #   arms in one invocation, or leave RUN_SUITE = None to run BASELINE_ARM.
    # =========================================================================
    BASELINE_ARM = "full"         # "full" | "B0" | "B1" | "B2" | "B3" | "B4" | "B5"
    RUN_SUITE    = None           # e.g. ["B0", "B1", "B2", "B3", "B4", "B5", "full"]
    CROPA_PROMPT_LEN = 8          # B4 (CroPA): number of soft prompt embeddings
    CROPA_PROMPT_LR  = 0.05       # B4 (CroPA): ascent step size on soft prompt embeddings

    # =========================================================================
    # v19 NOVEL IDEAS (Pruned 3-Part Core + 3 Principled Novel Additions)
    #   IDEA-N1: Soft Entity-Pair Spatial Envelope (detector-free gradient/attention
    #            localization of subject+object+relational corridor: full eps on
    #            foreground relational region, damped eps_bg = BGFG_RATIO * eps on
    #            un-attended background so scene objects never vanish and
    #            L_preserve <= C is satisfied without starving attack capacity).
    #   IDEA-N2: Bidirectional Converse Cycle-Consistency (enforces
    #            Belief(s, r_target, o) <=> Belief(o, r_converse, s) across open
    #            and yes/no training probes so the VLM updates its 3D scene
    #            geometry rather than memorizing a surface prefix on <subj>).
    #   IDEA-N3: Multi-Turn Conversational & Caption-Conditioned On-Policy Minting
    #            (mints 2-turn dialogue recovery rows conditioned on the clean
    #            scene caption and conversational follow-ups, directly closing
    #            the fup_eval and cap_eval held-out gap).
    #   PRUNE_KITCHEN_SINK: When True (in 'v19_core' and 'v19_novel'), zeroes out
    #            the legacy v14/v15 heuristic word-list penalties (OPT-A 33-word
    #            evasion list & OPT-C free-run heuristic) and disables inert
    #            CVaR/hardprobe/FMR/Lagrangian/RepCon scaffolding.
    # =========================================================================
    PRUNE_KITCHEN_SINK   = False  # True in 'v19_core' / 'v19_novel' presets
    NOVEL_SOFT_ENVELOPE  = True   # Idea N1: detector-free soft foreground/background eps envelope
    SOFT_ENV_FG_FRAC     = 0.70   # top 45% entity+corridor pixels get full 1.0 * eps
    SOFT_ENV_BG_RATIO    = 0.85   # remaining 55% background pixels capped at 0.35 * eps (~3.5/255)
    SOFT_ENV_SIGMA       = 6.0    # Gaussian smoothing radius to form a contiguous relational corridor
    NOVEL_CONVERSE_CYCLE = True   # Idea N2: bidirectional (o, r_converse, s) cycle-consistency probes
    CONVERSE_OPEN_COUNT  = 6      # number of converse open probes added to train pool
    CONVERSE_YN_COUNT    = 4      # number of converse yes/no frame pairs added to train pool
    NOVEL_MULTITURN_MINT = True   # Idea N3: multi-turn & caption-conditioned on-policy minting

    VERBOSE, VERBOSITY, LOG_LOSS_TERMS = True, 2, True

    IMAGES_DIR      = "/content/drive/MyDrive/VSR_Dataset/images"
    ANNOTATIONS_DIR = "/content/drive/MyDrive/VSR_Dataset/annotations"
    # v18 writes to its OWN directory. The eval probe set changed, so a v17
    # checkpoint is not comparable and must never be resumed into a v18 run.
    # ONE DIRECTORY PER ARM. The resume cache check (craft_phase, ~line 5007)
    # validates only eval_heldout, pool_split_seed and pool_reclaim_unused.
    # probe_lagrangian is NOT in that list, so pointing this arm at a previous
    # run's directory makes every example load from ck/ex_*.done.json, the
    # attack never runs, and you conclude the mechanism does nothing.
    # Rename this string for every arm (lagr_crn, famrw_crn, lagr_fast, ...).
    OUTPUT_DIR       = "/content/drive/MyDrive/v1812/all_fam_dicd_v2__seed0"
    DRIVE_MIRROR_DIR = None   # a shared mirror merges arms back together
    IO_RETRIES       = 5
    RESUME          = True

    MODEL_IDS  = ["llava-hf/llava-1.5-13b-hf"]
    LOAD_4BIT  = False
    IMAGE_SIZE = 336
    ATTN_IMPL  = "sdpa"
    BATCH_PROBES = True

    # =========================================================================
    # STEP 1 (v19): DATA SPLITS, OVERSAMPLING & SECOND BENCHMARK (What'sUp)
    #   Dev  = VSR "validation" (used during v15-v18.11 development)
    #   Test = VSR "test" (held-out; never looked at until final runs)
    #   Benchmark 2 = What'sUp (Kamath et al., EMNLP 2023) controlled images
    # =========================================================================
    BENCHMARK    = "vsr"          # "vsr" | "whatsup"
    SPLIT        = "test"         # "validation" (dev) | "test" (final) | "train"
    ALLOW_TRAIN_SUPPLEMENT = True # if VSR test split has <20 in a rare family,
                                  # pull the remaining rare-family items from VSR
                                  # train (never touching validation/dev!)
    NUM_EXAMPLES = 200            # Target n >= 200 test examples
    ONLY_TRUE    = True

    # Stratified family quotas / oversampling on test runs:
    # Guarantees vertical, proximity, and containment each reach >= 20 examples
    # instead of starving at n=3..5 while lateral takes 55% of the run.
    MIN_PER_FAMILY = {
        "lateral":     40,
        "depth":       40,
        "vertical":    20,
        "proximity":   20,
        "containment": 20,
    }
    MAX_PER_FAMILY = {            # caps dominant families until rare quotas fill
        "lateral":     80,
        "depth":       60,
        "vertical":    40,
        "proximity":   40,
        "containment": 40,
    }

    # What'sUp dataset paths / HuggingFace auto-download fallback
    WHATSUP_DIR     = "/content/drive/MyDrive/WhatsUp_Dataset"
    WHATSUP_HF_ID   = "kamath-amit/controlled_images"  # auto-fallback if local dir absent
    WHATSUP_SUBSET  = "controlled_images"              # "controlled_images" (A) | "controlled_clevr" (B)

    # =========================================================================
    # IDEA-3 (v18): paraphrase pool, disjoint eval split, worst-case DRO.
    # =========================================================================
    # PARAPHRASE_POOL=False restores the v17 3-variant bank AND the v17 eval
    # probe list exactly (including its train/eval overlap), for ablation.
    PARAPHRASE_POOL   = True
    POOL_SPLIT_SEED   = 1234   # fixed => the split is identical across examples,
                               # seeds, models and reruns. NEVER tune this.
    POOL_EVAL_FRAC    = 0.5    # half of each template family is held out
    # =========================================================================
    # STEP 2 (v19): EXPANDED HELD-OUT EVAL PROBES (15 probes total, 0 overlap)
    #   - 5 held-out open paraphrases (o19, o13, o4, o9, o16)
    #   - 3 held-out special open probes:
    #       [mc_eval]       held-out multiple-choice ("Which option best describes...")
    #       [wh_eval]       held-out direct "Where is X?" ("Where is the {s}? ...")
    #       [cap_eval]      held-out scene caption ("Write a one-sentence caption...")
    #   - 1 held-out 2-turn conversational follow-up:
    #       [fup_eval]      Turn 1 locates {o}, Turn 2 asks where {s} is relative to it
    #   - 3 held-out yes/no frames x {yes, no} = 6 polarity probes (y6/n6, y11/n11, y5/n5)
    #   Total = 5 + 3 + 1 + 6 = 15 held-out probes (while also logging the legacy
    #   7-probe subset `SBC_7` and `coherence_7` for direct comparison with Step 1).
    # =========================================================================
    EVAL_N_OPEN          = 5      # 5 open templates from held-out pool (was 3)
    EVAL_N_YESNO         = 3      # 3 frames x {yes,no} = 6 yes/no probes (was 2)
    EVAL_EXPANDED_PROBES = True   # adds mc_eval, wh_eval, cap_eval, fup_eval -> 15 total
    REEVAL_CACHED_IMAGES = True   # if a cached ex_*.done.json has 7 probes, load its
                                  # saved adv_png and evaluate the 15 probes in ~1 sec
                                  # without re-running 100 steps of APGD!

    # Stratified subset drawn per optimisation step. 6 -> 3 open + 1 yes + 2 no,
    # which is the same row count as one v17 variant, so step cost is unchanged.
    # v18.4: bumped 6 -> 8 (=> 4 open + 2 yes + 2 no) so MORE open frames see
    # gradient each step. The open family is where the held-out failures live, so
    # widening its per-step coverage is the augmentation route to closing the
    # train/eval gap. Cost rises ~8/6 = 1.33x per step. Set 10 for 5 open probes
    # (5 open + 2 yes + 3 no) if you can afford it; restore 6 for exact v18.3 cost.
    DRO_SUBSET        = 8

    # CVaR / worst-case weighting. Only the hardest ceil(alpha*n) probes in the
    # step's subset get gradient; the rest are dropped (weight 0, not forwarded).
    # alpha anneals 1.0 -> CVAR_ALPHA_FINAL so the EMA warms up on everything
    # first and no probe is starved before it has ever been scored.
    DRO_CVAR          = True
    # v18.4: 0.35 -> 1.0. CVaR narrowed gradient onto the hardest ceil(0.35*n)
    # TRAIN probes, but the failures are on DISJOINT held-out EVAL frames, so
    # worst-case reweighting was fitting a narrow train slice while the metric is
    # measured elsewhere (this file's own header: "the wrong tool when train and
    # eval are disjoint"). alpha=1.0 == uniform: every drawn probe gets gradient,
    # which is what a generalisation gap needs. With alpha=1.0 the anneal below is
    # a no-op (1.0 -> 1.0). Restore 0.35 to reproduce v18.3 worst-case behaviour.
    CVAR_ALPHA_FINAL  = 1.0
    CVAR_ANNEAL       = True

    # =========================================================================
    # v18.5 CVaR REWORK.  All four blocks default OFF, so PRESET="v18" is
    # bit-identical to v18.4.  Turn them on via the cvar_* presets.
    #
    # MOTIVATION (measured on the 25-example v18.4 alpha=1.0 run):
    #   12 examples fully coherent, 9 fail EXACTLY ONE probe, 4 fail broadly
    #   (those 4 own 9 of the 12 open failures).  A single global alpha cannot
    #   serve both regimes: the 9 single-fail examples want a narrow tail, the
    #   4 broad-fail ones want uniform coverage.  Fixing only the 9 would move
    #   coherence 0.48 -> 0.84 and SBC 0.874 -> 0.926.
    # =========================================================================

    # ---- (0) BUG FIXES in the v18.4 tail. Independent of the three ideas. ---
    # v18.4 ranked by ema.get(pkey, 0.0) DESCENDING, so a probe drawn for the
    # first time got the LOWEST possible key, was dropped, was never forwarded,
    # and its EMA stayed 0.0 forever. With 31 train probes and DRO_SUBSET=8 many
    # probes are first seen after the anneal has started cutting. Optimistic
    # init guarantees every probe is scored at least once.
    CVAR_COLDSTART_INF = True
    # Hard 0/1 cut quantises badly: k = ceil(alpha*n) with n=8 has only 8
    # reachable values, so alpha 0.26 and 0.35 are the SAME run. CVAR_MIX
    # interpolates between uniform (0.0) and the v18.4 hard cut (1.0) and gives
    # continuous control that alpha cannot. Mean weight stays LAMBDA_CONSIST.
    CVAR_MIX          = 1.0

    # ---- (i) ADAPTIVE RISK LEVEL -------------------------------------------
    # alpha stops being a hyperparameter and becomes an estimate of the tail
    # mass that actually exists: alpha_t = clip(fhat + delta, floor, 1) where
    # fhat is the decode-time failure fraction measured by the on-policy
    # rollout that already runs every ONPOLICY_EVERY steps. Broad-failure
    # examples get fhat ~ 0.5 -> alpha -> 1 (uniform coverage); single-fail
    # examples get fhat ~ 0.14 -> alpha -> floor (the binding probe gets 4x).
    # Requires ONPOLICY=True; falls back to CVAR_ALPHA_FINAL before the first
    # rollout.
    CVAR_ADAPTIVE     = False
    CVAR_ADAPT_DELTA  = 0.10   # headroom above the measured failure fraction
    CVAR_ALPHA_FLOOR  = 0.20   # never cut below this, even at fhat = 0
    CVAR_ADAPT_BETA   = 0.5    # EMA over triggers (rollouts are sparse + noisy)

    # ---- (ii) CALIBRATED RANKING -------------------------------------------
    # v18.4 ranked the tail by teacher-forced loss, but open-probe loss (~12
    # target tokens, margin+contrast+evasion active) and yes/no loss (~2 tokens,
    # CE fallback) are not on the same scale, so top-k partly selected a FAMILY
    # rather than a difficulty. Here a 2-parameter logistic is fit online from
    # the (loss, decode-success) pairs the rollout already produces, and the
    # tail is taken on predicted failure probability instead. This also replaces
    # the ONPOLICY_FAIL_BOOST=1.25 magic constant with a fitted quantity.
    CVAR_CALIBRATE    = False
    CVAR_CAL_LR       = 0.5    # SGD step on the logistic (log-loss)
    CVAR_CAL_MIN_OBS  = 8      # fall back to raw EMA until this many pairs seen
    CVAR_CAL_A0       = 1.0    # init slope on the STANDARDISED loss
    CVAR_CAL_B0       = 0.0    # init bias

    # ---- (iii) TYPED TAIL ---------------------------------------------------
    # Every open failure in the v18.4 run was `target_absent`; every yes/no
    # failure was `polarity`. Two distinct error modes that a scalar risk
    # variable cannot separate. With CVAR_TYPED the top-k is taken WITHIN each
    # family, so the open family can never be starved by boosted yes/no probes
    # (the mechanism that broke the alpha=0.35 run).
    CVAR_TYPED        = False

    # =========================================================================
    # v18.6 FAMILY FAILURE-MASS REWEIGHTING  (FMR).  Default OFF.
    #
    # THE MEASUREMENT THAT MOTIVATES IT.  Across every arm run so far the
    # failures are lopsided by FAMILY, not by individual probe:
    #
    #     family    share of train subset      share of held-out failures
    #     open      50%  (4 of 8)              ~80%  (9-10 of 12)
    #     yes/no    50%  (4 of 8)              ~20%
    #
    # Uniform weighting is uniform over PROBES, not over difficulty mass, so
    # the family that owns 80% of the errors receives 50% of the gradient.
    #
    # WHY NOT JUST USE CVaR.  A tail selects individuals: at alpha=0.35 with
    # DRO_SUBSET=8 only k=3 probes are funded, so the open family necessarily
    # receives FEWER than its 4 probes' worth of gradient. Every alpha<1 arm
    # therefore reduces coverage of exactly the family the held-out metric is
    # made of, which is why cvar_fix improved the training objective by 14%
    # and moved held-out SBC by zero. Tail selection cannot fix a group
    # allocation problem; it makes it worse.
    #
    # WHAT FMR DOES INSTEAD.  Keep FULL coverage -- every probe keeps a
    # non-zero weight, so nothing is ever starved -- and reallocate ACROSS
    # groups in proportion to each group's measured decode failure rate:
    #
    #     w_f  \propto  (p_f + eps) / sum_f' (p_f' + eps)
    #
    # p_f is estimated online from the on-policy rollout that already runs
    # every ONPOLICY_EVERY steps, so the allocation is measured rather than
    # assumed. This is a group reweighting with preserved support, NOT a risk
    # measure, and it is the only remaining variant whose predicted direction
    # agrees with everything measured so far.
    #
    # Composes with any base weighting (uniform, CVaR, softmax-EMA): FMR is a
    # multiplier applied to the base weights and then renormalised, so the mean
    # weight stays LAMBDA_CONSIST and the loss scale is comparable to every
    # other arm.
    # =========================================================================
    FAMILY_REWEIGHT   = False
    FAMILY_RW_EPS     = 0.05   # smoothing: a family at 0 observed failures
                               # still gets weight. Without this, a family that
                               # happens to pass every early rollout would be
                               # driven toward 0 -- the same starvation bug that
                               # CVAR_COLDSTART_INF exists to prevent.
    FAMILY_RW_BETA    = 0.5    # EMA over rollout triggers (they are sparse)
    FAMILY_RW_MIN     = 0.25   # clamp on the per-probe multiplier. Bounds the
    FAMILY_RW_MAX     = 4.0    # reallocation so no family can be starved and
                               # none can capture the whole budget.

    # =========================================================================
    # IDEA-2 (v18): on-policy prefix augmentation.
    # =========================================================================
    # v18.1: pressure raised. On the n=5 pilot the "next to" dodge survived on
    # bed/potted-plant across three probes, so every=20 / buffer=6 was too weak.
    ONPOLICY              = True
    ONPOLICY_EVERY        = 20     # steps between greedy roll-outs (cuts rollout & refresh overhead 2x)
    ONPOLICY_START_FRAC   = 0.20   # don't roll out before 20% of the budget
    ONPOLICY_OPEN_PROBES  = 4      # open probes rolled out per trigger
    ONPOLICY_YESNO_PROBES = 2      # yes/no probes rolled out per trigger
    ONPOLICY_GEN_TOKENS   = 12     # short decode: we only need the first clause
    ONPOLICY_PREFIX_TOKENS= 4      # how much of the wrong answer becomes context
    ONPOLICY_BUFFER       = 10     # FIFO capacity of minted rows (was 6)
    ONPOLICY_MAX_IN_BATCH = 3      # minted rows force-included in each step (was 2)
    ONPOLICY_FAIL_BOOST   = 1.25   # EMA boost for failing yes/no probes
    # v18.5. "ema" = v18.4 behaviour: a failing yes/no probe has its EMA
    # overwritten with max(all EMAs)*FAIL_BOOST, which puts it STRICTLY above
    # every other probe and guarantees it a tail slot. With alpha=0.35 and
    # DRO_SUBSET=8 the budget is k=3, and ONPOLICY_YESNO_PROBES=2 of those slots
    # could be captured this way -- by the family already at sbc_yes=1.000 --
    # while sbc_open collapsed 0.933 -> 0.667. "observe" records the failure for
    # the calibrator and the adaptive alpha WITHOUT hijacking the ranking.
    ONPOLICY_BOOST_MODE   = "ema"  # "ema" (v18.4) | "observe" (v18.5)
    # Minting rows changes the objective mid-run, which invalidates APGD's
    # f_best. When True the best-loss tracker is re-measured at the current
    # iterate right after each mint. Costs one extra fwd/bwd per trigger.
    ONPOLICY_REFRESH_BEST = True

    # =========================================================================
    # SCORER (v18.1). See the header note: the substring scorer failed in both
    # directions on the longer held-out answers.
    # =========================================================================
    SCORER_STRICT     = True   # clause-scoped + negation-aware. False = v18.0.
    SCORER_CLAUSE     = True   # scope the true-relation check to the subject's clause
    SCORER_NEGATION   = True   # a negated target match is NOT a hit
    DEGENERATE_CHECK  = True   # flag "the image does not show a bed" style answers
    SCORER_AUDIT      = True   # write the scorer's reason into probe_log rows

    # =========================================================================
    # v18.2 (a) MINT GATE. Only learn from genuine evasions.
    # =========================================================================
    ONPOLICY_MINT_GATE   = True
    MINT_REQUIRE_SUBJECT = True   # roll-out must name the subject
    MINT_REQUIRE_OBJECT  = False  # ... naming the object too is often too strict
    MINT_REJECT_DEGEN    = True   # never mint from "the image does not show X"

    # =========================================================================
    # v18.2 (b) SPATIAL MASK. Freeze most pixels; edit only where it matters.
    # =========================================================================
    # MASK_KEEP_FRAC is the Pareto knob. 1.0 == v18.1 (no mask).
    # OFF by default ON PURPOSE. The mask trades SBC for fidelity; enabling it
    # silently would make PRESET="v18" incomparable to the v18.1 run. Opt in via
    # PRESET="ablate_mask" or PRESET="curve_mask".
    MASK_ENABLE        = False
    MASK_MODE          = "saliency"   # "saliency" | "none"
    MASK_KEEP_FRAC     = 0.15         # fraction of pixels allowed to change
    MASK_BLUR_SIGMA    = 4.0          # smooth saliency into regions, not speckle
    MASK_DILATE        = 9            # max-pool radius (odd); 0 disables
    MASK_REFRESH_EVERY = 0            # 0 = compute once at x_clean; >0 = re-derive
    MASK_MIN_PIXELS    = 256          # never shrink below this many editable px

    # =========================================================================
    # v18.2 (c) CURVE MODE. Record the whole curve, never the best point.
    # =========================================================================
    CURVE_MODE        = "none"        # "none" | "eps" | "mask"
    EPS_CURVE_VALUES  = [2/255, 4/255, 8/255, 12/255, 16/255]
    MASK_CURVE_VALUES = [0.05, 0.15, 0.30, 0.60, 1.0]
    # Which point supplies the top-level entry metrics, so results.json stays
    # comparable to earlier runs. Must be a member of the active curve list.
    CURVE_REFERENCE   = 16/255        # for CURVE_MODE="eps"; ignored otherwise

    # =========================================================================
    # v18.3 (A) PRESERVATION LOSS. Optimise the thing we report.
    # =========================================================================
    # Every run since v15 reported preserve_strict=0.000 and preserve_f1 ~0.65
    # while NOTHING in the objective touched either. Labels are the CLEAN
    # model's own greedy answers to PRESERVATION_QUESTIONS, frozen once before
    # the attack, so this is a pure self-consistency term: no external caption
    # source, no extra annotation, and it is exactly the quantity evaluate_full
    # scores afterwards.
    PRESERVE_LOSS        = True
    PRESERVE_QUESTIONS_N = 2      # how many of PRESERVATION_QUESTIONS to use
    PRESERVE_MAX_TOKENS  = 40     # cap on the frozen clean label length
    # Fixed-weight mode (PRESERVE_LAGRANGIAN=False) uses PRESERVE_WEIGHT
    # directly. It is the honest ablation baseline but needs hand-tuning.
    PRESERVE_WEIGHT      = 0.5    # also the initial lambda in Lagrangian mode
    # Constrained mode:  min L_lie  s.t.  L_preserve <= PRESERVE_C.
    # PRESERVE_C is in nats/token of the frozen clean answer. ~0.5 is a loose
    # constraint (the caption drifts but survives); ~0.1 is tight. SWEEP THIS
    # to draw the Pareto curve -- it is the v18.3 replacement for the eps and
    # mask curves, and unlike them each point is a controlled trade-off.
    PRESERVE_LAGRANGIAN  = True
    PRESERVE_C           = 0.35
    PRESERVE_LAMBDA_LR   = 0.05   # dual step. Small: lambda moving makes the
                                  # objective non-stationary, same as minting.
    PRESERVE_LAMBDA_MAX  = 8.0    # hard cap; without it a hard constraint can
                                  # run away and swamp the lie objective
    PRESERVE_LAMBDA_MIN  = 0.0
    # Re-measure APGD's f_best when lambda has moved materially, for the same
    # reason ONPOLICY_REFRESH_BEST exists: the old best refers to a loss that
    # no longer exists. Costs one extra fwd/bwd per trigger.
    PRESERVE_REFRESH_BEST      = True
    PRESERVE_REFRESH_THRESHOLD = 0.25   # relative lambda change that triggers it
    # Curve over the constraint level. "none" | "preserve_c"
    PRESERVE_CURVE_VALUES = [0.10, 0.20, 0.35, 0.60, 1.00]

    # =========================================================================
    # v18.7 IDEA-A: PER-PROBE LAGRANGIAN CONSTRAINT SATISFACTION.
    # =========================================================================
    # WHY. The reported metric (coherence) is a CONJUNCTION over held-out
    # probes: an example scores 1 only if EVERY probe is satisfied. Every loss
    # term in this file is a weighted SUM over probes. A sum has no reason to
    # lift the weakest probe once the mean is low -- which is exactly what the
    # v18.6 results show: 69% of (example, probe) pairs never changed under any
    # objective edit, and coherence sat at 0.20-0.30 while SBC sat at ~0.78.
    #
    # CVaR, HARDPROBE_WEIGHTING and FAMILY_REWEIGHT are three different soft
    # approximations to "attend to the worst probe". This replaces all three
    # with the constrained problem they are approximating:
    #
    #     min ||delta||   s.t.   L_p(x+delta) <= PROBE_C   for every probe p
    #                            L_preserve   <= PRESERVE_C
    #
    # solved by dual ascent, one multiplier per probe:
    #
    #     lambda_p <- clip(lambda_p + LR * (L_p - PROBE_C), 0, LAMBDA_MAX)
    #
    # A satisfied probe's price DECAYS toward zero and the budget moves to the
    # violated ones automatically. No alpha to tune, no family partition, no
    # softmax temperature.
    #
    # THE SECOND PAYOFF -- INFEASIBILITY CERTIFICATES. A probe whose multiplier
    # saturates at LAMBDA_MAX for a sustained fraction of the run while its
    # loss stays above PROBE_C has been paid the maximum admissible price and
    # still not been bought. That is a measured statement that the constraint
    # is not satisfiable at this eps, not a silent failure. The five probes
    # that failed in 5/5 arms of the v18.6 sweep should certify as infeasible;
    # if they do not, the diagnosis was wrong and that is worth knowing too.
    PROBE_LAGRANGIAN      = False   # master switch. False == exact v18.6.
    PROBE_C               = 7.5    # per-probe loss budget (nats-ish, same
                                    # units as the softmin row loss). Sweeping
                                    # it draws a coherence-vs-eps Pareto curve
                                    # on the CONSTRAINT rather than on capacity.
    PROBE_LAMBDA_INIT     = 1.0     # == LAMBDA_CONSIST, so step 0 is bit-equal
                                    # to the uniform arm.
    PROBE_LAMBDA_LR       = 0.15    # dual step. Larger than PRESERVE_LAMBDA_LR
                                    # because there are many duals and each one
                                    # sees a much noisier constraint value.
    # NOT 0.0. loss_and_grad DROPS any probe whose weight is 0 (the CVaR skip
    # path), so a probe whose price decays to exactly zero stops being
    # forwarded, stops being measured, its _probe_last entry goes stale, its
    # dual freezes and it can never come back even after the perturbation
    # drifts and re-violates it. That is precisely the silent-starvation bug
    # the v18.5 header documents for CVaR cold-start, re-introduced through a
    # different door. A small floor keeps every constraint OBSERVED at
    # negligible cost -- and a constraint you stopped measuring is not a
    # constraint you are enforcing.
    PROBE_LAMBDA_MIN      = 0.05
    PROBE_LAMBDA_MAX      = 8.0     # the price ceiling that defines "certified
                                    # infeasible". Raise it and fewer probes
                                    # certify; that is the intended knob.
    PROBE_LAMBDA_DECAY    = 0.0     # optional pull toward INIT on satisfied
                                    # probes; 0 = pure dual ascent.
    # Normalise the weights actually applied to the loss so their MEAN stays
    # LAMBDA_CONSIST. The RAW multipliers are kept separately for the
    # certificates. Without this, a run where every constraint is violated
    # inflates the lie objective relative to the preservation term and the two
    # duals start fighting over an arbitrary scale rather than a trade-off.
    PROBE_LAMBDA_NORMALIZE = True
    # A probe is certified infeasible if its RAW multiplier sat within
    # CERT_TOL of LAMBDA_MAX for >= CERT_FRAC of the scored steps AND its final
    # loss is still above PROBE_C.
    PROBE_CERT_FRAC       = 0.50
    PROBE_CERT_TOL        = 0.02
    # v18.8. A constraint value older than this many dual steps is STALE and
    # cannot support a certificate. With DRO_SUBSET=8 out of ~31-48 probes, most
    # probes are not redrawn on any given step, so without this the certificate
    # is dominated by probes that simply were not looked at. 10 is roughly one
    # full expected sweep of the pool at DRO_SUBSET=8; raise it if `coverage` in
    # the certificate block comes back low.
    PROBE_STALE_MAX       = 10
    # Re-measure APGD's f_best after a material change in the weight vector,
    # for the same reason PRESERVE_REFRESH_BEST exists.
    PROBE_REFRESH_BEST      = True
    # Relative L1 drift of the weight vector SINCE THE LAST REFRESH (not since
    # the last step -- see the note in update_probe_lambdas). 0.35 was far too
    # high once measured correctly; 0.10 fires roughly 5-10 times per 60-step
    # run, which is what keeping f_best valid requires.
    PROBE_REFRESH_THRESHOLD = 0.10
    # Infeasibility demotion. Off by default so it is a separate ablation.
    PROBE_DEMOTE_INFEASIBLE = False
    PROBE_DEMOTE_TO         = 1.0    # price cap once a probe certifies
    PROBE_DEMOTE_MIN_STEPS  = 15     # do not judge a probe on 3 noisy steps
    _UNUSED_OLD_THRESHOLD   = 0.35   # recorded for provenance
    PROBE_LAMBDA_EVERY      = 1      # dual steps per APGD iteration
    PROBE_LAG_DEBUG         = True   # per-step telemetry (see _dbg_probe_lag)
    PROBE_LAG_DEBUG_EVERY   = 10
    PROBE_LAG_LOG_TRACE     = True   # persist the full multiplier trace to JSON
    PROBE_C_CURVE_VALUES    = [0.25, 0.50, 0.75, 1.25, 2.00]

    # =========================================================================
    # IDEA-C: REPRESENTATION-SPACE BELIEF CONSISTENCY (repcon.py).
    #   Attacks the residual stream: pushes each probe's relation-decision
    #   hidden state across a target-vs-true axis and minimises its variance
    #   across phrasings. Targets the open/no held-out failures that per-probe
    #   allocation cannot fix. CHECK AFTER THE RUN: the "[repcon] ON ... sep="
    #   line (sep must be non-tiny) and repcon.frac_on_target in results.json.
    # =========================================================================
    REPCON_ENABLE       = False
    REPCON_LAYER        = 20     # decoder layer to tap (13B LM has 40; sweep)
    REPCON_WEIGHT       = 1.0
    REPCON_GAMMA        = 1.0    # invariance term weight relative to steer
    REPCON_MARGIN_FRAC  = 1.0    # steer target = this * clean(target-vs-true) sep
    REPCON_N_DIR_PROBES = 4      # #train templates used to build the direction
    REPCON_DIR_TARGET   = "object"  # IDEA#1: steer OBJECT-word spatial-ID (Kang'26)
                                 # "object" | "subject" | "relation" (old, nulled)
    REPCON_DEBUG        = True    # verbose repcon telemetry (direction + attack)
    # =========================================================================
    # LOSS_MODE: counterfactual-teacher (mirror distillation) loss.
    #   "classic"    -> the original _row_loss (default; all old results).
    #   "cf_teacher" -> REPLACE _row_loss by KL(teacher || attacked), where the
    #                   teacher is the model's own next-token distribution on
    #                   the HORIZONTALLY MIRRORED clean image (on which the
    #                   left/right relation really is the target).
    #   "cf_hybrid"  -> classic + CF_WEIGHT * KL.
    # Only families in CF_FAMILIES use it (mirroring only swaps left/right);
    # all other examples fall back to "classic". Run mirror_check.py first.
    # =========================================================================
    LOSS_MODE           = "cf_hybrid"  # v18.9 default (was "classic"); set
                                  # "classic" for the A0 control arm.
    ONLY_FAMILIES       = None    # None = attack all 5 families; set ["lateral"]
                                  # for lateral-only runs.
    CF_FAMILIES         = ["lateral", "vertical", "depth", "containment", "proximity"]
    # Counterfactual operator selection:
    #   "auto"   -> horizontal mirror on lateral (bit-identical to v18.11) +
    #               Dual-Inversion (T_prop + T_swap) on vertical, depth,
    #               containment, proximity.
    #   "dual"   -> force clean-only DICD on ALL families (including
    #               lateral) for direct A/B comparison against mirror.
    #   "mirror" -> legacy horizontal mirror only (lateral only).
    CF_COUNTERFACTUAL_MODE = "auto"
    CF_WEIGHT           = 1.0     # KL weight in cf_hybrid
    CF_TEMP             = 1.0     # distillation temperature
    CF_REQUIRE_GATE     = True    # use the teacher only if, on the mirror, the
                                  # clean model says Yes to target & No to truth
    CF_KINDS            = ["yes", "no"]  # question types that use the teacher.
                                  # mirror_check: open answers are "next to" on
                                  # BOTH images (no left/right info), so open
                                  # questions keep the classic loss.
    CF_DEBUG            = True
    CF_OPEN_DIR         = True   # direction-only mirror teacher on OPEN rows: KL over
                                  # {true dir, target dir} only, weighted per position by
                                  # the teacher's direction mass (mirror_probe verdict A)
    CF_OPEN_WEIGHT      = 0.3    # v18.8: was 1.0. The open-dir KL is ADDED onto the
                                 # classic row loss, so at 1.0 it competed with (and
                                 # broke) examples the classic attack already solved.
                                 # Kept small so it can nudge but never dominate.
    CF_OPEN_DIR_MIN_MASS = 0.30  # v18.8: percept gate. Turn the open-dir teacher ON
                                 # only if the mirror actually PERCEIVES a direction,
                                 # i.e. puts >= this much probability on {left,right}
                                 # after "The <subj> is to the". A positive dir_gap on
                                 # a diffuse "next to" distribution (low dir_mass) is a
                                 # noisy target that hurt more than it helped, so
                                 # dir_gap>0 alone is NOT sufficient.
    # --- v18.8 PGCM (Perception-Gated Counterfactual Margin), Stage 1 + Stage 2 ---
    # Redesign of the open-dir mirror teacher so that turning it on can NOT drop
    # accuracy step by step. Two ideas replace the old two-sided KL:
    #   Stage 1 (percept trust beta): a CONTINUOUS weight in [0,1] built from the
    #     mirror's own direction mass + gap. beta -> 0 exactly on the no-percept
    #     "next to" scenes, so the term silences itself smoothly (no hard cliff).
    #   Stage 2 (one-sided hinge): instead of matching the mirror's blurry
    #     distribution, only push logp(target) above logp(true) by a margin m0 the
    #     mirror's OWN confidence sets. The hinge is one-sided and saturates, so
    #     once the student already prefers the target by m0 the gradient is ~0 and
    #     the term can never pull a solved example back (this is what broke
    #     train/potted plant under the KL).
    CF_PGCM             = True    # False -> fall back to the legacy open-dir KL.
    CF_PGCM_MASS_C0     = 0.30    # beta: centre of the dir_mass sigmoid (percept)
    CF_PGCM_MASS_S      = 0.10    # beta: slope of the dir_mass sigmoid
    CF_PGCM_GAP_S       = 1.50    # beta: slope of the dir_gap sigmoid (logits)
    CF_PGCM_MARGIN_MAX  = 4.0     # hinge: cap on the teacher-anchored margin m0
    # --- v18.10 MBPD diagnostic (measurement only; see file header) ---
    MBPD_DIAG           = True    # 112 no-grad forwards per LATERAL example
    MBPD_MARGIN         = 1.0     # m: logit margin a stated lie must clear
    MBPD_P_MIN          = 0.5     # |P| below this (swing < 1.0) = perception_dead
    MBPD_ALPHA_MAX      = 8.0     # alpha above this is reported as "capped"
    # --- v18.11 MBPD LOSS (see file header). Requires LOSS_MODE != classic and
    # CF_CONTRAST=True for part (1); part (2) only needs the CF teacher active.
    MBPD_LOSS           = True    # False -> exactly v18.10 (the A2 arm)
    MBPD_ALPHA_CAP      = 3.0     # max per-frame extrapolation used IN the loss
    MBPD_FRAG_W         = 0.65    # calibrated extra weight (1.0 -> up to 1.65x) on fragile rows
    MBPD_FRAG_P0        = 0.75    # P at which the boost is half-on
    MBPD_FRAG_S         = 0.25    # slope of the boost sigmoid
    # --- High-assent / high-dissent coherence refinements ---
    MBPD_SHORTFALL_WEIGHT    = True   # drive fragility by max(P0 - P_f, -sgn*g_cf)
                                      # so high-bias frames get maximum boost
    MBPD_SYMMETRIC_FRAG      = True   # also boost 'Yes' rows that are beyond_mirror /
                                      # perception_dead under dissent frames (y2/y9/y11)
    MBPD_DEAD_ASSENT_TEACHER = True   # keep CCT on with capped alpha when a row
                                      # is perception_dead but has strong opposing clean
                                      # bias and P_f > 0.05 (cancels frame prior)
    MBPD_DEAD_P_FLOOR        = 0.05   # min positive P_f to allow dead-row teacher
    STRATIFIED_NO_BIAS       = False  # keep balanced 4 open + 2 yes + 2 no sampling
    YESNO_PIVOT_FIRST_TOKEN  = True   # pmask=[True, False] on 'Yes.'/'No.' so the
                                      # period token does not halve token-0 margin
    # --- v18.9 CCT: Contrastive Counterfactual Teacher (see file header) ---
    CF_CONTRAST         = True    # False -> v18.8 plain mirror-KL teacher (A1 arm)
    CF_CONTRAST_ALPHA   = 1.0     # a in z* = (1+a) z_mirror - a z_clean (0 = mirror only)
    CF_CONTRAST_BETA    = 0.1     # plausibility head: keep tokens with
                                  # p_mirror >= BETA * max p_mirror (VCD constraint)
    CF_CONTRAST_JS      = True    # per-position JS(mirror||clean) weighting
    CF_CONTRAST_JS_REF  = 0.15    # JS (nats, max ln2=0.693) that earns full weight 1
    CF_CONTRAST_REPLACE_GATE = True  # yes/no teacher runs even if the hard gate
                                  # fails; the JS weight silences it on
                                  # no-percept scenes. False -> keep hard gate.
    CF_CONTRAST_OPEN    = True    # open-dir PGCM: beta also uses the contrastive
                                  # swing dir_gap(mirror) - dir_gap(clean)
    CF_CONTRAST_SWING_C0 = 1.0    # swing (logits) at which the swing trust = 0.5
    CF_CONTRAST_SWING_S  = 0.75   # slope of the swing sigmoid
    OPEN_DIAG           = False   # after each LATERAL example: classify every open-
                                  # question failure as dodge vs wrong direction
    CF_DEBUG_EVERY      = 20      # print KL telemetry every N loss calls
    REPCON_FAMILIES     = ["lateral"]  # METHOD-1: only steer these families;
                                 # None = steer all (old behaviour). Default steers
                                 # only lateral (the one family the spatial-ID helped).
    REPCON_SCAN         = False  # if True, print a clean-image layer scan on the
                                 # first valid example (pick the best tap layer)
    REPCON_SCAN_LAYERS  = [8, 12, 16, 20, 24, 28, 32, 36]
    REPCON_PATCH_PROBE  = False  # IDEA-3: causal patching probe (clean vs adv)
                                 # after each example. Diagnostic only, no attack
                                 # change. Prints per-layer causal depth of the lie.
    REPCON_PATCH_LAYERS = None   # None -> auto (every 3rd layer); or e.g. list(range(0,40,2))
    REPCON_DECOUPLE     = False  # measure output-flip vs logit-gap decoupling per
                                 # example (the "bed" phenomenon). Diagnostic only.

    # =========================================================================
    # v18.7 IDEA-B: COMMON RANDOM NUMBERS (paired stochastic streams).
    # =========================================================================
    # WHY. Three v18.6 arms (results 79 / 81 / 82) were the SAME config at the
    # SAME seed and returned SBC 0.771 / 0.757 / 0.786, with per-example swings
    # of +-0.286 -- the same magnitude as every effect being attributed to an
    # objective change. The cause is that the random init and the EOT dither
    # stream are consumed at positions that depend on how many times
    # loss_and_grad happens to be called, and THAT depends on the arm (mint
    # refreshes, dual refreshes, CVaR skipping every probe, ...). Two arms
    # therefore ride different noise trajectories and the paired delta measures
    # the trajectory, not the objective.
    #
    # FIX. Make every stochastic draw a pure function of
    #       (example key, restart salt, phase, step index, draw index)
    # and NOT of the stream position. Then arm A and arm B see bit-identical
    # dither tensors at step k, bit-identical random init, and bit-identical
    # probe subsets -- so the paired difference isolates the objective. This is
    # common random numbers / paired sampling, standard in simulation and
    # absent from every attack-ablation table we could find.
    #
    # CRN_STRICT_PROBES additionally pins the per-step probe subset draw, which
    # matters because CVaR and the Lagrangian can otherwise consume the global
    # RNG a different number of times per step.
    CRN_ENABLE         = True
    CRN_STRICT_PROBES  = True
    CRN_STREAM_SEED    = 0x5EED      # change to get an independent paired block
    CRN_DEBUG          = True        # print a fingerprint of the drawn streams
    # Fingerprints let you PROVE two arms shared the streams instead of hoping.
    # Each is a cheap float summary of the first dither and the random init.
    CRN_FINGERPRINT_STEPS = (0, 1, 10, 25, 50)
    # Kernel-level determinism, applied INSIDE the attack process (see the
    # block near [init] CUDA). Notebook-level torch settings do not reach here,
    # because run_arm.py spawns a fresh interpreter per arm. Costs some speed.
    DETERMINISTIC      = False   # False enables A100 TF32 Tensor Cores, cuDNN
                                  # benchmark, and fast Flash/SDPA backward kernels
                                  # while CRN_ENABLE=True keeps all random streams pinned.

    # =========================================================================
    # v18.8 (0) POOL RECLAIM.  THE DEAD-TEMPLATE BUG.
    # =========================================================================
    # _pool_split() sized the held-out halves as
    #     n_y = max(EVAL_N_YESNO, round(14 * POOL_EVAL_FRAC)) = 7
    # but build_eval_probes consumes only POOL["yn_eval"][:EVAL_N_YESNO] = 2.
    # The other 5 frames were in NEITHER set: never optimised, never measured.
    # Same arithmetic on the open side leaves 7 of 20 base templates dead.
    #
    # MEASURED CONSEQUENCE. The train pool ran at ~64% of the prompt diversity
    # already written in this file, on a problem diagnosed (v18 header) as a
    # paraphrase GENERALISATION gap. Reclaiming them adds 5 frames x 2
    # polarities + 7 open = 17 train probes.
    #
    # THE EVAL SET IS UNTOUCHED. Only the first EVAL_N_* indices were ever
    # evaluated and they keep their positions, so held-out SBC stays comparable
    # with every run since v18.0. _assert_eval_split_stable() enforces this at
    # import time rather than trusting the claim.
    #
    # NOTE it changes the TRAINING distribution, so this is not a drop-in
    # reproduction of a v18.4/v18.7 training config -- it is a new arm.
    POOL_RECLAIM_UNUSED = False

    # =========================================================================
    # v18.8 (1) POLARITY AT THE OBJECTIVE.
    # =========================================================================
    # MEASUREMENT THAT MOTIVATES IT (20 example-runs, 140 held-out probes):
    #
    #     frame                       yes-probes      no-probes
    #     "Can you confirm that..."   20/20 = 1.000   10/20 = 0.500
    #     "Would it be accurate..."   19/20 = 0.950   15/20 = 0.750
    #
    # On laptop/cake the model answered "Yes." to BOTH polarities of the SAME
    # confirm frame while all three open probes and the other yes/no frame
    # stated the implanted lie correctly. So the belief IS implanted; one frame
    # simply fails to query it. That is assent bias, not a belief failure.
    #
    # Two consequences this block acts on:
    #   (a) yes-probes are carried by the prior and no-probes fight it, so a
    #       SYMMETRIC contrast margin under-serves exactly the failing family.
    #   (b) raising the MEAN no-probe margin over train frames does not
    #       transfer, because assent is frame-specific. Penalise the SPREAD.
    CONTRAST_MARGIN_NO = 9.0    # calibrated (9.0 vs 6.0 on 'yes') to clear high-assent
                                # confirmation frames ('Can you confirm...' n6) without overshoot.
    # Frame-invariance penalty on the no-family. A perturbation that satisfies
    # "Is it correct that..." but not "Can you confirm..." has HIGH variance
    # across no-frames even when the mean is low; a sum objective is blind to
    # that and this is not. Needs >= 2 no-probes in the step's subset to bite --
    # with DRO_SUBSET=8 the stratifier gives ~2, so raise DRO_SUBSET if the
    # telemetry shows it firing rarely.
    POLARITY_INVAR   = False
    POLARITY_INVAR_W = 0.5

    # =========================================================================
    # v18.8 (2) HONEST MEASUREMENT.  Nothing here enters the gradient.
    # =========================================================================
    # sbc_yes = 0.975 is NOT evidence the attack works if the clean model also
    # assents. Score the same 7 held-out probes on the CLEAN image and report
    # the LIFT. Costs one extra forward pass per probe per example.
    EVAL_CLEAN_CONTROL = True
    # Per-template/per-frame accuracy, so a failure can be attributed to a
    # frame rather than averaged away. Requires the pkey tag added to
    # build_eval_probes.
    EVAL_PER_FRAME     = True

    # =========================================================================
    # v18.8 (3) ADAPTIVE PER-EXAMPLE BUDGET.
    # =========================================================================
    # WHY. Coherence is a CONJUNCTION over 7 probes. Across 20 example-runs:
    #     0 misses   4 examples   (coherence = 0.20)
    #     1 miss     8 examples   <- one probe from perfect
    #     2+ misses  8 examples
    # Fixing only the 8 single-miss examples moves coherence 0.20 -> 0.60 and
    # SBC 0.743 -> 0.800. By SBC MASS the hard examples dominate (78% of
    # misses); by COHERENCE LEVERAGE the near-miss ones do. Every example
    # currently gets the same STEPS regardless, so the optimiser cannot act on
    # the distinction. This reallocates steps ACROSS examples.
    #
    # THE HONESTY CONSTRAINT. The decision reads _op_fail_frac, measured by the
    # on-policy rollout on TRAIN probes only. Conditioning step count on EVAL
    # probes would be selection on the test metric -- the thing the v18.2 header
    # forbids. Never wire this to eval.
    #
    # COST NEUTRALITY. Steps are drawn from a shared pool fed by early exits and
    # caps, so total compute tracks the baseline. granted != returned at the end
    # of a run means this arm spent more than its baseline and the comparison is
    # VOID; that is printed, not hidden.
    BUDGET_ADAPT       = False
    BUDGET_DECIDE_FRAC = 0.50   # decide at this fraction of the base steps
    BUDGET_LOW         = 0.20   # fail_frac <= LOW  -> near-coherent, EXTEND
    BUDGET_HIGH        = 0.50   # fail_frac >= HIGH -> broad failure, CAP
    BUDGET_BONUS       = 40     # max extra steps granted to one example
    BUDGET_MAX_MULT    = 2.0    # hard ceiling on n_iter
    BUDGET_EARLY_EXIT  = True   # fail_frac == 0 -> stop and return the remainder
    BUDGET_DEBUG       = True

    # =========================================================================
    # v18.3 (B) CE / MARGIN SEPARATION.
    # =========================================================================
    # CE is unbounded and never switches off; the CW hinge stops at MARGIN.
    # Applying both to a pivot token means the perturbation keeps growing long
    # after the argmax is decided, which is what put hallucinated objects into
    # the v18.2 adversarial captions. Split them by position instead.
    # Falls back to full-sequence CE when the pivot mask is all-True (yes/no),
    # so no probe is left with a zero objective.
    CE_NONPIVOT_ONLY = True

    # =========================================================================
    # v18.3 (C) yes/no polarity contrast, now ON by default.
    # =========================================================================
    # OPT-F lives in _V16_FLAGS, which PRESET="v18" clears wholesale. This flag
    # re-enables it AFTER that clear, so v15/v16/v17 stay bit-reproducible.
    V183_YESNO_CONTRAST = True

    # =========================================================================
    # v18.3 (D) APGD checkpoint condition 1.
    # =========================================================================
    # "prev" = Croce & Hein (improvement over the previous iterate).
    # "best" = the v18.2 behaviour (improvement over the running best), which
    #          is strictly harder to satisfy and halves eta prematurely.
    APGD_COND1_MODE = "prev"
    # Iterations whose objective just changed (mint / lambda step) are excluded
    # from the improvement count rather than scored against a stale loss.
    APGD_SKIP_CHANGED_STEPS = True

    # =========================================================================
    # v18.3 (E) BUDGET.
    # =========================================================================
    # Multiplies every entry of STEPS_BY_FAMILY. 1.0 keeps v18.2 comparability.
    # Try 4.0 on ONE example (PRESET="steps_probe") and read LOSS_HISTORY: if
    # the loss is still falling at 4x, the failures were under-convergence; if
    # it plateaus, the target is infeasible at that eps -- which is itself the
    # publishable result.
    STEPS_SCALE      = 1.0
    LOG_LOSS_HISTORY = True    # persist the APGD loss curve into results.json

    # --- attack ---
    EPS            = 10 / 255
    STEPS          = 100     # default; overridden per-family by STEPS_BY_FAMILY (OPT4)
    RANDOM_INIT    = True
    LAMBDA_CONSIST = 1.0     # base per-probe objective weight (mean weight under OPT-B)
    PROMPT_EOT     = True    # sample a paraphrase variant each step

    # --- IDEA 1 (kept): multi-reference soft-min + forced-choice alignment ---
    MULTI_REF        = True
    ADD_FORCED_CHOICE= True
    MAX_OPEN_REFS    = 4
    SOFTMIN_TAU      = 4.0

    # --- OPT1 (v12): pivot-token CW margin (target vs argmax competitor) ---
    MARGIN_WEIGHT    = 1.0
    CE_WEIGHT        = 1.0
    MARGIN           = 6.0
    PIVOT_ONLY       = True

    # --- OPT1 (v13): contrastive pivot margin (target vs TRUE relation word) ---
    CONTRAST_WEIGHT  = 1.0
    CONTRAST_MARGIN  = 6.0

    # --- OPT-A (v14 NEW): evasion / off-axis suppression -----------------------
    # Keep the correct label token above these escape words at EVERY answer
    # position of open probes. Built from v13's observed failure log + a complete
    # off-axis spatial set. The per-row isin guard exempts the target's own words
    # (so "in front of" targets never suppress "front"). Watch preserve_f1: if it
    # drops below ~0.60, lower EVASION_WEIGHT before touching anything else.
    EVASION_WEIGHT   = 0.5        # 0 -> off (exact v13 behavior)
    EVASION_MARGIN   = 4.0        # per-position gap: z_label - z_evasion
    EVASION_WORDS    = [
        "next", "beside", "nearby", "near", "close", "adjacent",
    ]

    # --- OPT-B (v14 NEW): detached hard-probe weighting (targets coherence) -----
    HARDPROBE_WEIGHTING = True    # False -> uniform weights == exact v13
    HARDPROBE_TAU       = 0.5     # softmax temperature over per-probe EMA difficulty
    HARDPROBE_BETA      = 0.9     # EMA decay of per-probe loss (higher = smoother)

    # --- OPT-C (v15): free-run divergence margin -------------------------------
    # Attack the model's REAL greedy branch point: the first teacher-forced
    # position whose argmax != target IS the token greedy decoding emits wrong
    # first. Add a GENTLE margin there so the model cannot start a dodge /
    # off-axis / true-relation / wrong-yes-no answer. Applies to ALL probe kinds.
    #
    # IMPORTANT (v15 fix): the first v15 run used fr_w=2.0 / fr_margin=8.0, which
    # made this ONE unnormalized term dominate the loss (~5-10x the others),
    # drowned out the pivot/contrast terms that actually produced the lie, and
    # REGRESSED working examples to SBC=0. The term is now gentle: weight <= the
    # other margin weights, margin <= the evasion margin. Keep it that way.
    # v16 RETUNED TO 0.0. Measured on the 20-example pilot (same seeds, best-of-3
    # sweep) OPT-C is monotonically HARMFUL as a mechanism:
    #     fr=0.00  SBC .800  coh .200  open .733  no .725
    #     fr=0.25  SBC .743  coh .200  open .700  no .700
    #     fr=0.50  SBC .664  coh .100  open .600  no .550
    # The headline 0.836 came from max-over-3-runs, NOT from free-run. So v16
    # keeps the best-of-N machinery but spends the budget on RESTARTS at the
    # strongest single setting instead (see SWEEP_MODE below).
    # v18.4: 0.5 -> 0.25. Bring the free-run divergence margin BACK, but gently.
    # It suppresses the "next to"/"under"/"near" dodge WHERE THE MODEL ACTUALLY
    # EMITS IT (the free-run branch token), which teacher-forced evasion never
    # scores. The header found OPT-C harmful only when it DOMINATED the loss
    # unnormalised (fr>=0.5); at 0.25 it is <= the other margin weights, so it
    # nudges the open branch without drowning the pivot/contrast terms that
    # actually produce the lie. This is now the OPERATIVE value: FREERUN_SWEEP
    # below is set to a single [0.25], so free-run is always on rather than being
    # one of three swept values that reverted to 0.0 on the easy examples.
    FREERUN_WEIGHT   = 0.25       # v18.4 operative value (was 0.5)
    FREERUN_MARGIN   = 4.0        # <= evasion margin

    # --- OPT-C ablation sweep (SAFE DEFAULT) -----------------------------------
    # If non-empty, each example is attacked once per value here (same seed), and
    # the BEST result (by SBC, then coherence, then preserve_f1) is kept. Because
    # 0.0 == exact v14, the default sweep can NEVER do worse than your v14
    # baseline: worst case fr=0 wins, best case OPT-C helps — and you get a
    # per-example comparison table. Cost: len(sweep)x compute per example.
    # To run a single fixed value instead, set FREERUN_SWEEP = [] (uses
    # FREERUN_WEIGHT above).
    # v18.4: [0.0, 0.25, 0.5] -> [0.25]. The old best-of-3 sweep kept whichever
    # value scored highest per example, which on the depth winners was fr=0.0 --
    # i.e. free-run was effectively OFF exactly where the scene was easy, and
    # never got a chance to fix the open dodges elsewhere. A single [0.25] applies
    # the gentle free-run margin to EVERY example, and also cuts compute 3x (one
    # run per example, not three).
    #   TRADE-OFF, stated plainly: dropping 0.0 removes the "never worse than the
    #   fr=0 baseline" safety net the header valued. If you want the safety net
    #   AND gentle free-run, use [0.0, 0.25] (best-of-2). To reproduce v18.3
    #   exactly, restore [0.0, 0.25, 0.5].
    FREERUN_SWEEP    = [0.25]

    # =========================================================================
    # v16 ADDITIONS -- ALL DEFAULT OFF IN v17.  Measured, not theorised:
    #
    # Controlled A/B (same 3 examples, same seed, same fr=0.0, so the ONLY
    # difference is the four loss/bank flags below):
    #     bowl/banana   0.857 -> 1.000   (+0.143)
    #     person/cow    0.857 -> 0.714   (-0.143)
    #     cow/sheep     0.857 -> 0.286   (-0.571)
    #     mean          0.857 -> 0.667   (-0.190)
    #
    # n=3 is small, but the direction is consistent and the comparison is clean.
    # Do NOT switch several of these on together again -- v16 changed four
    # things at once, which is why the regression could not be attributed.
    # =========================================================================

    # --- OPT-D: 'Does X appear ...' training frame ---------------------------
    # RATIONALE THAT DID NOT SURVIVE CONTACT WITH DATA. In v15, 'appear'-framed
    # no-probes scored 10/20 while 'is it correct'-framed ones scored 20/20, and
    # only the latter was in the training bank. Adding the missing frame made it
    # WORSE (2/3 -> 1/3 on the rerun). So the frame gap was correlational, not
    # causal: "Does X appear <true relation>?" appears to be intrinsically hard
    # -- 'appear' cues the model toward its visual evidence -- and putting the
    # string in the bank does not change that.
    # CONFOUND, if you retest: a 4th variant means PROMPT_EOT gives each variant
    # 25% of the steps instead of 33%. Set COMPENSATE_STEPS=True or the test
    # measures budget dilution rather than the frame.
    ADD_APPEAR_FRAME = False

    # --- OPT-F: yes/no polarity contrast -------------------------------------
    # Still believe the observation is real (v15 left forbid_ids=None on every
    # yes/no probe, so the contrast term never ran there). But it adds a term to
    # an already-crowded objective and was never tested alone.
    YESNO_CONTRAST   = False

    # --- OPT-G: multi-reference yes/no answers --------------------------------
    # Suspect. Soft-min picks the EASIEST reference, so offering "Yes, it is."
    # lets the optimiser satisfy the probe with a hedged template instead of a
    # bare "Yes." -- and eval generates greedily and free-form.
    MULTI_REF_YESNO  = False
    MAX_YESNO_REFS   = 2

    # --- OPT-E: filler evasion words ------------------------------------------
    # 'located'/'situated' appear in FAILING answers ("is located next to") but
    # also in SUCCEEDING ones ("The bowl is located to the right of"), so
    # suppressing them may cost as much as it gains.
    ADD_FILLER_EVASION = False

    # --- OPT-H: sweep mode ----------------------------------------------------
    # "freerun"  v15 default: sweep FREERUN_SWEEP values, keep best. <- DEFAULT
    #            The v15 sweep's value was HETEROGENEITY, not its mean. fr=0.5
    #            averaged worst (0.664 vs 0.800) yet won on 7/20 examples,
    #            because different settings fail on different examples. Worth
    #            +0.036 over always running fr=0.0. v16 replaced it with
    #            same-config restarts and threw that decorrelation away.
    # "restart"  N seeds at FREERUN_WEIGHT.
    # "hybrid"   fr values x restarts (costs len(sweep)*RESTART_SWEEP runs).
    # "off"      single run.
    SWEEP_MODE       = "freerun"
    RESTART_SWEEP    = 3
    # Stop a sweep once an example is already perfect. Cannot change the kept
    # result (perfect cannot be beaten) -- it only truncates the sweep table, so
    # set False if you want the full per-value comparison for a paper appendix.
    SWEEP_EARLY_EXIT = True

    # --- v17: hold per-variant optimisation budget constant --------------------
    # PROMPT_EOT samples ONE variant per step, so adding a variant silently cuts
    # every variant's share of STEPS. When True, STEPS is scaled by
    # n_variants/3 so an added frame is tested on its merits, not on budget.
    COMPENSATE_STEPS = True

    # --- OPTION 3 (v12): straight-through 8-bit quant + EOT ---
    QUANT_STE        = True
    EOT_SAMPLES      = 2          # default EOT draws; overridden by EOT_SAMPLES_BY_FAMILY
    EOT_DITHER       = 1.0 / 255
    QUANT_PROJECT    = True

    # --- OPT4 (v13, DIALED BACK in v14): family budget + antithetic EOT ---------
    STEPS_BY_FAMILY = {           # dialed back near 100 so n>=300 is runnable
        "lateral":     100,       # small bump; lateral is still the hardest axis
        "depth":       100,
        "vertical":    100,
        "containment": 100,
        "proximity":   100,
        "other":       100,
    }
    EOT_SAMPLES_BY_FAMILY = {}    # empty -> every family uses EOT_SAMPLES (2)
    EOT_ANTITHETIC   = True       # cheap variance reduction; keep on

    DEBUG_ATTACK     = True
    RUN_RANDOM_CONTROL = True

    APGD_RHO, APGD_ALPHA_MOM, APGD_ETA_INIT_MULT = 0.75, 0.75, 2.0
    GEN_MAX_NEW_TOKENS = 40
    SEED = 0

    # --- v12 MEMORY & SPEED CONTROLS ---
    # Raised to 32 with Prefix KV-Caching so all ~25 active probe + preserve rows
    # run in ONE batched forward+backward pass even on 16GB/24GB GPUs.
    MAX_ROWS_PER_FWD = 32
    PREFIX_KV_CACHE  = True       # run 580+ image prefix tokens ONCE (batch_size=1,
                                  # use_cache=True), expand past_key_values to B, and
                                  # run the 13B LM ONLY on the ~35 text tail tokens
    TWO_STAGE_EOT    = True       # use T=1 EOT draw for the first EOT_STAGE2_FRAC of
                                  # steps (when eta >> 1 LSB) and full T=2 antithetic
                                  # EOT in the final polish stage
    EOT_STAGE2_FRAC  = 0.70       # switch from 1 EOT draw to EOT_SAMPLES at 70% progress
    VISION_DEDUP     = True       # run CLIP vision tower on batch_size=1 when all
                                  # B rows share the same image, then expand to B
    FAST_TOKENIZE    = True       # skip redundant CPU CLIPImageProcessor in encode_probe
    BATCH_GENERATE   = True       # batch greedy generation in _onpolicy_update & eval
    GRAD_CHECKPOINT  = False
    EMPTY_CACHE_EVERY = 0


FILLER_EVASION_WORDS = ["located", "situated"]

_V16_FLAGS = ("ADD_APPEAR_FRAME", "YESNO_CONTRAST", "MULTI_REF_YESNO",
              "ADD_FILLER_EVASION")

_V18_FLAGS = ("PARAPHRASE_POOL", "DRO_CVAR", "ONPOLICY")

# v18.2 flags. Cleared by the v15/v16/v17 presets so old runs stay reproducible.
_V182_FLAGS = ("ONPOLICY_MINT_GATE", "MASK_ENABLE")

# v18.3 flags. Same contract: every older preset must clear these, or "v17"
# stops meaning v17 and the ablation ladder silently stops being a ladder.
_V183_FLAGS = ("PRESERVE_LOSS", "CE_NONPIVOT_ONLY", "V183_YESNO_CONTRAST")

_ABLATIONS = {
    # --- v18 ablations. "v15"/"v16"/"v17" force ALL v18 flags off, so the old
    # numbers stay reproducible bit-for-bit.
    # Each of these isolates ONE mechanism. They must therefore also switch the
    # v18.2 flags OFF explicitly, or they silently inherit the CFG defaults and
    # stop being single-variable ablations.
    "ablate_pool":       {"PARAPHRASE_POOL": True,  "DRO_CVAR": False, "ONPOLICY": False,
                          "ONPOLICY_MINT_GATE": False, "MASK_ENABLE": False},
    "ablate_cvar":       {"PARAPHRASE_POOL": True,  "DRO_CVAR": True,  "ONPOLICY": False,
                          "ONPOLICY_MINT_GATE": False, "MASK_ENABLE": False},
    "ablate_onpolicy":   {"PARAPHRASE_POOL": False, "DRO_CVAR": False, "ONPOLICY": True,
                          "ONPOLICY_MINT_GATE": False, "MASK_ENABLE": False},
    # --- v18.2 ---
    "ablate_mintgate":   {"PARAPHRASE_POOL": True, "DRO_CVAR": True, "ONPOLICY": True,
                          "ONPOLICY_MINT_GATE": True, "MASK_ENABLE": False},
    "ablate_mask":       {"PARAPHRASE_POOL": True, "DRO_CVAR": True, "ONPOLICY": True,
                          "ONPOLICY_MINT_GATE": True, "MASK_ENABLE": True},
    "curve_eps":         {"PARAPHRASE_POOL": True, "DRO_CVAR": True, "ONPOLICY": True,
                          "ONPOLICY_MINT_GATE": True, "MASK_ENABLE": False,
                          "CURVE_MODE": "eps", "SWEEP_MODE": "off",
                          "SWEEP_EARLY_EXIT": False},
    "curve_mask":        {"PARAPHRASE_POOL": True, "DRO_CVAR": True, "ONPOLICY": True,
                          "ONPOLICY_MINT_GATE": True, "MASK_ENABLE": True,
                          "CURVE_MODE": "mask", "SWEEP_MODE": "off",
                          "SWEEP_EARLY_EXIT": False},
    # --- v18.3 ---------------------------------------------------------------
    # (i) The loss-term ablations that were never run. Five hand-weighted terms
    #     with no ablation evidence was the weakest part of the write-up.
    #     Each turns exactly ONE term off from the full v18.3 objective.
    "ablate_noce":       {"CE_WEIGHT": 0.0},
    "ablate_nomargin":   {"MARGIN_WEIGHT": 0.0},
    "ablate_nocontrast": {"CONTRAST_WEIGHT": 0.0},
    "ablate_noevasion":  {"EVASION_WEIGHT": 0.0},
    "ablate_nofreerun":  {"FREERUN_WEIGHT": 0.0, "FREERUN_SWEEP": [],
                          "SWEEP_MODE": "off"},
    # (ii) CE/margin separation, isolated.
    "ablate_ce_split":   {"CE_NONPIVOT_ONLY": True},
    "ablate_ce_nosplit": {"CE_NONPIVOT_ONLY": False},
    # (iii) Preservation term, isolated. "fixed" is the honest baseline for the
    #       Lagrangian: same term, hand-tuned constant weight.
    "ablate_nopreserve": {"PRESERVE_LOSS": False},
    "ablate_preserve":   {"PRESERVE_LOSS": True, "PRESERVE_LAGRANGIAN": True},
    "ablate_preserve_fixed": {"PRESERVE_LOSS": True, "PRESERVE_LAGRANGIAN": False,
                              "PRESERVE_WEIGHT": 0.5},
    # (iv) CVaR alpha. PREDICTION recorded before running: cvar_100 generalises
    #      best to the held-out probes; cvar_035 wins on train-seen ones only.
    "cvar_035":          {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 0.35},
    "cvar_060":          {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 0.60},
    "cvar_100":          {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 1.00},
    # --- v18.5 CVaR rework. RUN THE LADDER IN THIS ORDER. Each arm adds
    #     exactly one mechanism to the one above it, so a gain is attributable.
    #     cvar_100 is the control: alpha=1.0 makes k=ceil(1.0*n)=n, keep=all,
    #     w=n*LAMBDA/n=LAMBDA -- arithmetically identical to uniform.
    # (a) v18.4 tail + the two bug fixes ONLY. Tests the hypothesis that
    #     alpha=0.35 lost because of budget capture, not because tail
    #     optimisation is wrong.
    "cvar_fix":          {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 0.35,
                          "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    # (b) soft tail: continuous control that alpha cannot give at n=8.
    "cvar_soft":         {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 0.35,
                          "CVAR_MIX": 0.5, "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    # (c) IDEA (i) alone: alpha measured from the decode failure fraction.
    "cvar_adapt":        {"DRO_CVAR": True, "CVAR_ADAPTIVE": True,
                          "CVAR_ALPHA_FINAL": 0.35, "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    # (d) IDEA (ii) alone: rank the tail by calibrated P(fail), not by loss.
    "cvar_cal":          {"DRO_CVAR": True, "CVAR_CALIBRATE": True,
                          "CVAR_ALPHA_FINAL": 0.35, "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    # (e) IDEA (iii) alone: one tail per family.
    "cvar_typed":        {"DRO_CVAR": True, "CVAR_TYPED": True,
                          "CVAR_ALPHA_FINAL": 0.35, "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    # --- v18.6 FAMILY FAILURE-MASS REWEIGHTING -----------------------------
    # (g) THE PROPOSAL. Uniform base (alpha=1 -> full coverage, nothing
    #     starved) + reallocation across families by measured failure rate.
    #     Compare against cvar_100, which is the SAME arm with FMR off.
    "famrw":             {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 1.00,
                          "FAMILY_REWEIGHT": True},
    # (h) FMR composed with the fixed tail. Tests whether group reallocation
    #     and individual tail selection are complementary or redundant.
    "famrw_cvar":        {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 0.35,
                          "FAMILY_REWEIGHT": True,
                          "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    "famrw_floor":       {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 1.00,
                          "FAMILY_REWEIGHT": True,
                          "FAMILY_RW_MIN": 1.0, "FAMILY_RW_MAX": 4.0},
    # (i) sensitivity: how hard is the reallocation allowed to push?
    "famrw_soft":        {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 1.00,
                          "FAMILY_REWEIGHT": True,
                          "FAMILY_RW_MIN": 0.5, "FAMILY_RW_MAX": 2.0},
    # (f) all three composed. This is the v18.5 proposal.
    "cvar_v185":         {"DRO_CVAR": True, "CVAR_ADAPTIVE": True,
                          "CVAR_CALIBRATE": True, "CVAR_TYPED": True,
                          "CVAR_ALPHA_FINAL": 0.35, "CVAR_MIX": 1.0,
                          "CVAR_COLDSTART_INF": True,
                          "ONPOLICY_BOOST_MODE": "observe"},
    # (v) APGD cond1, isolated. "apgd_cond1_best" reproduces v18.2 exactly.
    "apgd_cond1_prev":   {"APGD_COND1_MODE": "prev"},
    "apgd_cond1_best":   {"APGD_COND1_MODE": "best"},
    # (vi) Budget diagnostic. Run on ONE example and read the loss history.
    "steps_probe":       {"STEPS_SCALE": 4.0, "SWEEP_MODE": "off",
                          "FREERUN_SWEEP": [], "NUM_EXAMPLES": 1,
                          "LOG_LOSS_HISTORY": True},
    # (vii) The Pareto curve, drawn on the CONSTRAINT instead of on capacity.
    "curve_preserve":    {"PRESERVE_LOSS": True, "PRESERVE_LAGRANGIAN": True,
                          "CURVE_MODE": "preserve_c", "SWEEP_MODE": "off",
                          "SWEEP_EARLY_EXIT": False},
    "ablate_appear":     {"ADD_APPEAR_FRAME": True},
    "ablate_yesno_ctr":  {"YESNO_CONTRAST": True},
    "ablate_yesno_refs": {"MULTI_REF_YESNO": True},
    "ablate_evasion":    {"ADD_FILLER_EVASION": True},
    "ablate_restart":    {"SWEEP_MODE": "restart"},
    # =====================================================================
    # v18.7 IDEA-A: per-probe Lagrangian. RUN THIS LADDER IN ORDER.
    # =====================================================================
    # The control. `famrw` is the v18.6 proposal; `lagr` is the SAME arm with
    # the allocation rule swapped. Both normalise to mean weight
    # LAMBDA_CONSIST, so any difference is the allocation, not the scale.
    "lagr":              {"PROBE_LAGRANGIAN": True,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    # Dual step sensitivity. If `lagr` shows lambda_cv < 0.05 the duals are too
    # slow to differentiate inside the step budget -- try this before
    # concluding the mechanism does not work.
    "lagr_fast":         {"PROBE_LAGRANGIAN": True, "PROBE_LAMBDA_LR": 0.40,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    # Raw multipliers applied directly. Tests whether the normalisation (which
    # buys scale comparability) is costing the mechanism its bite.
    "lagr_raw":          {"PROBE_LAGRANGIAN": True,
                          "PROBE_LAMBDA_NORMALIZE": False,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    # Certificate sensitivity: a higher ceiling means fewer probes certify
    # infeasible. Report the certificate count as a FUNCTION of this, never as
    # a single number.
    "lagr_ceiling":      {"PROBE_LAGRANGIAN": True, "PROBE_LAMBDA_MAX": 20.0,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    # The Pareto curve drawn on the PER-PROBE constraint. Each point is a
    # controlled trade-off, not a capacity throttle like eps or the mask.
    "curve_probe_c":     {"PROBE_LAGRANGIAN": True, "CURVE_MODE": "probe_c",
                          "SWEEP_MODE": "off", "SWEEP_EARLY_EXIT": False,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    # =====================================================================
    # v18.7 IDEA-B: common random numbers. These two arms exist to MEASURE
    # the noise floor, not to win. Run each over >= 3 salts and compare the
    # spread: crn_off should show the +-0.03 arm-level / +-0.29 per-example
    # spread seen in results 79/81/82; crn_on should collapse it.
    # =====================================================================
    "crn_on":            {"CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "crn_off":           {"CRN_ENABLE": False, "CRN_STRICT_PROBES": False},
    # The 2x2 that separates the two noise channels. Streams vs kernels.
    # det_off isolates how much of the residual is cudnn.benchmark + tf32;
    # det_on is the fully-pinned condition. Run both paired.
    "det_on":            {"CRN_ENABLE": True, "CRN_STRICT_PROBES": True,
                          "DETERMINISTIC": True},
    "det_off":           {"CRN_ENABLE": True, "CRN_STRICT_PROBES": True,
                          "DETERMINISTIC": False},
    # The headline pairing: IDEA-A measured under IDEA-B.
    # v18.7b. The refresh fix applies to every lagr arm automatically; these
    # two isolate the demotion change on top of it.
    "lagr_demote":       {"PROBE_LAGRANGIAN": True, "PROBE_DEMOTE_INFEASIBLE": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    "lagr_refresh_off":  {"PROBE_LAGRANGIAN": True, "PROBE_REFRESH_BEST": False,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    "lagr_crn":          {"PROBE_LAGRANGIAN": True, "CRN_ENABLE": True,
                          "CRN_STRICT_PROBES": True,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False},
    "famrw_crn":         {"DRO_CVAR": True, "CVAR_ALPHA_FINAL": 1.00,
                          "FAMILY_REWEIGHT": True, "CRN_ENABLE": True,
                          "CRN_STRICT_PROBES": True},

    # =====================================================================
    # v18.8 LADDER. RUN IN THIS ORDER. Each arm adds exactly ONE mechanism
    # to `base88`, so a gain is attributable. All of them sit on the full
    # v18.3/v18.7 objective (see the _v183_baseline whitelist below) --
    # otherwise they would silently also disable preservation, the pool and
    # on-policy, and "reclaim vs base" would be measuring five changes.
    # =====================================================================
    # (0) THE CONTROL. v18.7 behaviour plus measurement only. Run this FIRST:
    #     it costs one extra forward pass per probe and it tells you how much
    #     of your existing SBC is the model's assent prior rather than the
    #     attack. Every later arm is judged on SBC_lift, not SBC.
    "base88":            {"EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (1) Reclaim the 12 dead templates. Eval set byte-identical; the TRAIN
    #     distribution grows by 17 probes. Largest expected effect.
    "reclaim":           {"POOL_RECLAIM_UNUSED": True,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (1b) Reclaim + wider per-step coverage. The pool grows 31 -> 48 while
    #      DRO_SUBSET stays at 8, so each probe is drawn ~35% less often. This
    #      arm separates "diversity helped" from "coverage hurt". Costs ~1.5x
    #      per step.
    "reclaim_wide":      {"POOL_RECLAIM_UNUSED": True, "DRO_SUBSET": 12,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (2) Asymmetric polarity margin. Cheap, and a useful NULL: if this alone
    #     closes the no-probe gap then the failure was margin size, not frame
    #     generalisation, and (3) is unnecessary.
    "polmargin":         {"CONTRAST_MARGIN_NO": 10.0,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (3) Frame-invariance penalty on the no-family. Check
    #     polarity_invar_fires in results.json BEFORE reading its SBC.
    "polinvar":          {"POLARITY_INVAR": True,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (3b) The variance term needs >= 2 no-probes in one micro-batch. If
    #      `polinvar` shows fires ~ 0, this is the arm to run instead of
    #      raising POLARITY_INVAR_W.
    "polinvar_wide":     {"POLARITY_INVAR": True, "DRO_SUBSET": 12,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (4) Adaptive per-example budget. JUDGE THIS ON COHERENCE, NOT SBC: it
    #     trades a small SBC gain for a large coherence gain by construction.
    #     Scoring it on SBC is how `famrw` was misread.
    "budget":            {"BUDGET_ADAPT": True,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # (5) Everything composed. Only meaningful once the single-mechanism arms
    #     above have been read; running this first produces a number nobody can
    #     attribute to anything.
    "v188":              {"POOL_RECLAIM_UNUSED": True, "POLARITY_INVAR": True,
                          "CONTRAST_MARGIN_NO": 10.0, "BUDGET_ADAPT": True,
                          "DRO_SUBSET": 12,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # =====================================================================
    # STEP 3 BASELINES (Publication Plan Section 4 & Section 5 Step 3)
    #   All baselines run on the exact same VSR test split, same eps (10/255),
    #   same 100 steps, same 15 held-out eval probes, and same CRN seed.
    # =====================================================================
    "B0":                {"BASELINE_ARM": "B0", "POOL_RECLAIM_UNUSED": True,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "B1":                {"BASELINE_ARM": "B1", "POOL_RECLAIM_UNUSED": True,
                          "LOSS_MODE": "classic", "ONPOLICY": False,
                          "PRESERVE_LOSS": False, "CE_NONPIVOT_ONLY": False,
                          "MARGIN_WEIGHT": 0.0, "CONTRAST_WEIGHT": 0.0,
                          "EVASION_WEIGHT": 0.0, "FREERUN_WEIGHT": 0.0,
                          "FREERUN_SWEEP": [0.0], "MULTI_REF": False,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "B2":                {"BASELINE_ARM": "B2", "POOL_RECLAIM_UNUSED": True,
                          "LOSS_MODE": "classic", "ONPOLICY": False,
                          "PRESERVE_LOSS": False, "CE_NONPIVOT_ONLY": False,
                          "MARGIN_WEIGHT": 0.0, "CONTRAST_WEIGHT": 0.0,
                          "EVASION_WEIGHT": 0.0, "FREERUN_WEIGHT": 0.0,
                          "FREERUN_SWEEP": [0.0], "MULTI_REF": False,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "B3":                {"BASELINE_ARM": "B3", "POOL_RECLAIM_UNUSED": True,
                          "LOSS_MODE": "cf_teacher", "CF_CONTRAST": False,
                          "MBPD_LOSS": False, "CF_OPEN_DIR": False,
                          "CF_KINDS": ["open", "yes", "no"],
                          "ONPOLICY": False, "PRESERVE_LOSS": False,
                          "CE_NONPIVOT_ONLY": False, "MARGIN_WEIGHT": 0.0,
                          "CONTRAST_WEIGHT": 0.0, "EVASION_WEIGHT": 0.0,
                          "FREERUN_WEIGHT": 0.0, "FREERUN_SWEEP": [0.0],
                          "MULTI_REF": False,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "B4":                {"BASELINE_ARM": "B4", "POOL_RECLAIM_UNUSED": True,
                          "LOSS_MODE": "classic", "ONPOLICY": False,
                          "PRESERVE_LOSS": False, "CE_NONPIVOT_ONLY": False,
                          "MARGIN_WEIGHT": 0.0, "CONTRAST_WEIGHT": 0.0,
                          "EVASION_WEIGHT": 0.0, "FREERUN_WEIGHT": 0.0,
                          "FREERUN_SWEEP": [0.0], "MULTI_REF": False,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "B5":                {"BASELINE_ARM": "B5", "POOL_RECLAIM_UNUSED": True,
                          "LOSS_MODE": "classic", "ONPOLICY": False,
                          "PRESERVE_LOSS": False,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    # =====================================================================
    # v19 CLEANED & NOVEL ARMS
    #   "v19_core"  : Pruned 3-part core ONLY (CE/Pivot Margin + On-Policy
    #                 Minting + CCT/DICD + Preservation). Removes OPT-A
    #                 (33-word evasion list), OPT-C (free-run heuristic),
    #                 CVaR, hard-probe EMA, FMR, RepCon.
    #   "v19_novel" : Pruned 3-part core + ALL 3 NOVEL IDEAS:
    #                 (N1) Soft Entity-Pair Spatial Envelope
    #                 (N2) Bidirectional Converse Cycle-Consistency
    #                 (N3) Multi-Turn & Caption-Conditioned On-Policy Minting
    # =====================================================================
    "v19_core":          {"BASELINE_ARM": "v19_core", "POOL_RECLAIM_UNUSED": True,
                          "PRUNE_KITCHEN_SINK": True,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False, "PROBE_LAGRANGIAN": False,
                          "REPCON_ENABLE": False,
                          "EVASION_WEIGHT": 0.0, "FREERUN_WEIGHT": 0.0,
                          "FREERUN_SWEEP": [0.0],
                          "NOVEL_SOFT_ENVELOPE": False,
                          "NOVEL_CONVERSE_CYCLE": False,
                          "NOVEL_MULTITURN_MINT": False,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
    "v19_novel":         {"BASELINE_ARM": "v19_novel", "POOL_RECLAIM_UNUSED": True,
                          "PRUNE_KITCHEN_SINK": True,
                          "DRO_CVAR": False, "HARDPROBE_WEIGHTING": False,
                          "FAMILY_REWEIGHT": False, "PROBE_LAGRANGIAN": False,
                          "REPCON_ENABLE": False,
                          "EVASION_WEIGHT": 0.35, "EVASION_MARGIN": 4.5,
                          "FREERUN_WEIGHT": 0.20, "FREERUN_SWEEP": [0.20],
                          "DRO_SUBSET": 12,
                          "CONTRAST_MARGIN": 7.5, "CONTRAST_MARGIN_NO": 10.0,
                          "MBPD_FRAG_W": 0.85,
                          "PRESERVE_C": 0.55, "PRESERVE_LAMBDA_LR": 0.03,
                          "PRESERVE_LAMBDA_MAX": 1.75,
                          "ONPOLICY_EVERY": 15, "ONPOLICY_START_FRAC": 0.15,
                          "ONPOLICY_BUFFER": 12, "ONPOLICY_MAX_IN_BATCH": 4,
                          "NOVEL_SOFT_ENVELOPE": True,
                          "NOVEL_CONVERSE_CYCLE": True,
                          "NOVEL_MULTITURN_MINT": True,
                          "EVAL_CLEAN_CONTROL": True, "EVAL_PER_FRAME": True,
                          "CRN_ENABLE": True, "CRN_STRICT_PROBES": True},
}


def _apply_preset():
    """Resolve CFG.PRESET into concrete flags. Runs once at import."""
    p = getattr(CFG, "PRESET", "custom")
    if p == "custom":
        pass
    elif p in ("v18", "v18.3"):
        for f in _V16_FLAGS:
            setattr(CFG, f, False)
        for f in _V18_FLAGS:
            setattr(CFG, f, True)
        # v18 default = v18.1 + the mint-gate bug fix. The mint gate can only
        # reduce wasted budget, so it is safe to enable by default. The mask is
        # NOT, because it costs SBC by design.
        CFG.ONPOLICY_MINT_GATE = True
        CFG.MASK_ENABLE = False
        CFG.CURVE_MODE = "none"
        CFG.SWEEP_MODE = "freerun"
        # --- v18.3 defaults. All three are objective-side and all three are
        # justified by a measured failure, so they are ON here. Use
        # PRESET="ablate_no<x>" to isolate any of them.
        CFG.PRESERVE_LOSS      = True
        CFG.CE_NONPIVOT_ONLY   = True
        CFG.V183_YESNO_CONTRAST = True
        CFG.APGD_COND1_MODE    = "prev"
    elif p in ("v15", "v17", "v18.2"):
        for f in _V16_FLAGS:
            setattr(CFG, f, False)
        for f in _V18_FLAGS + _V182_FLAGS + _V183_FLAGS:
            setattr(CFG, f, False)
        CFG.CURVE_MODE = "none"
        CFG.SWEEP_MODE = "freerun"
        CFG.APGD_COND1_MODE = "best"      # the v18.2 cond1, reproduced exactly
        if p == "v18.2":
            # Exact v18.2 semantics: v18 ideas ON, v18.3 objective changes OFF.
            for f in _V18_FLAGS:
                setattr(CFG, f, True)
            CFG.ONPOLICY_MINT_GATE = True
    elif p == "v16":
        for f in _V16_FLAGS:
            setattr(CFG, f, True)
        for f in _V18_FLAGS + _V182_FLAGS + _V183_FLAGS:
            setattr(CFG, f, False)
        CFG.CURVE_MODE = "none"
        CFG.SWEEP_MODE = "restart"
        CFG.FREERUN_WEIGHT = 0.0
        CFG.APGD_COND1_MODE = "best"
        CFG.COMPENSATE_STEPS = False      # v16 did NOT compensate -- that is the bug
    elif p in _ABLATIONS:
        for f in _V16_FLAGS:              # start from a clean v15 baseline
            setattr(CFG, f, False)
        # These presets isolate ONE mechanism against the full v18.3 objective,
        # so they must NOT have the v18/v18.2/v18.3 flags cleared underneath
        # them -- otherwise "ablate_nocontrast" would be measuring four changes.
        _self_setting = ("ablate_pool", "ablate_cvar", "ablate_onpolicy",
                         "ablate_mintgate", "ablate_mask", "curve_eps", "curve_mask")
        _v183_baseline = ("ablate_noce", "ablate_nomargin", "ablate_nocontrast",
                          "ablate_noevasion", "ablate_nofreerun",
                          "ablate_ce_split", "ablate_ce_nosplit",
                          "ablate_nopreserve", "ablate_preserve",
                          "ablate_preserve_fixed",
                          "cvar_035", "cvar_060", "cvar_100",
                          "cvar_fix", "cvar_soft", "cvar_adapt", "cvar_cal",
                          "cvar_typed", "cvar_v185",
                          "famrw", "famrw_cvar", "famrw_soft", "famrw_floor",
                          "apgd_cond1_prev", "apgd_cond1_best",
                          "steps_probe", "curve_preserve",
                          # --- v18.7. All of these must sit on the FULL
                          # v18.3/v18.6 objective, because they change exactly
                          # one thing: the probe allocation rule (IDEA-A) or
                          # the noise stream (IDEA-B). If they fell through to
                          # the `else` branch below they would silently also
                          # disable preservation, the pool and on-policy, and
                          # "lagr vs famrw" would be measuring five changes.
                          "lagr", "lagr_fast", "lagr_raw", "lagr_ceiling",
                          "curve_probe_c", "lagr_crn", "famrw_crn",
                          "crn_on", "crn_off", "det_on", "det_off",
                          "lagr_demote", "lagr_refresh_off",
                          # --- v18.8. These change ONE thing each: the prompt
                          # pool, the polarity objective, or the step
                          # allocation. If they fell through to the `else`
                          # branch they would also silently disable
                          # preservation, the paraphrase pool and on-policy.
                          "base88", "reclaim", "reclaim_wide", "polmargin",
                          "polinvar", "polinvar_wide", "budget", "v188",
                          "B0", "B1", "B2", "B3", "B4", "B5",
                          "v19_core", "v19_novel")
        if p in _v183_baseline:
            # full v18.3 objective, then the dict turns exactly one thing off
            for f in _V18_FLAGS:
                setattr(CFG, f, True)
            CFG.ONPOLICY_MINT_GATE = True
            CFG.MASK_ENABLE = False
            for f in _V183_FLAGS:
                setattr(CFG, f, True)
            CFG.CURVE_MODE = "none"
        elif p in _self_setting:
            for f in _V183_FLAGS:         # v18-era ablations predate v18.3
                setattr(CFG, f, False)
            CFG.APGD_COND1_MODE = "best"
        else:
            for f in _V18_FLAGS + _V182_FLAGS + _V183_FLAGS:
                setattr(CFG, f, False)
            CFG.CURVE_MODE = "none"
            CFG.APGD_COND1_MODE = "best"
        if p not in ("curve_eps", "curve_mask", "curve_preserve", "curve_probe_c"):
            CFG.SWEEP_MODE = "freerun"
        for k, v in _ABLATIONS[p].items():
            setattr(CFG, k, v)
    else:
        raise ValueError(f"unknown PRESET {p!r}; valid: v15, v16, v17, v18, custom, "
                         + ", ".join(sorted(_ABLATIONS)))

    if CFG.ADD_FILLER_EVASION:
        for w in FILLER_EVASION_WORDS:
            if w not in CFG.EVASION_WORDS:
                CFG.EVASION_WORDS.append(w)

    # --- v18.3 (C). YESNO_CONTRAST is a _V16_FLAG, so every modern preset has
    # just cleared it. Re-assert it here, AFTER the clear, so v15/v16/v17 stay
    # bit-reproducible while v18.3 gets the term the data asked for. Setting it
    # inside each preset branch instead would be four places to forget.
    if getattr(CFG, "V183_YESNO_CONTRAST", False):
        CFG.YESNO_CONTRAST = True

    # --- v18.3 (E). One knob for the whole family budget.
    _sc = float(getattr(CFG, "STEPS_SCALE", 1.0) or 1.0)
    if _sc != 1.0 and not getattr(CFG, "_STEPS_SCALED", False):
        CFG._STEPS_SCALED = True
        CFG.STEPS = max(10, int(round(CFG.STEPS * _sc)))
        CFG.STEPS_BY_FAMILY = {k: max(10, int(round(v * _sc)))
                               for k, v in CFG.STEPS_BY_FAMILY.items()}
        print(f"[preset] STEPS_SCALE={_sc} -> STEPS_BY_FAMILY={CFG.STEPS_BY_FAMILY}")

    if CFG.PRESERVE_LOSS and not CFG.PRESERVE_LAGRANGIAN and CFG.PRESERVE_WEIGHT <= 0:
        print("[preset] WARNING: PRESERVE_LOSS is on with weight 0 and no dual "
              "update. The term will be measured but never optimised, which is "
              "the v18.2 situation this release exists to fix.")

    on = [f for f in _V16_FLAGS if getattr(CFG, f)]
    on18 = [f for f in _V18_FLAGS if getattr(CFG, f)]
    print(f"[preset] PRESET={p!r} -> v16 flags ON: {on or 'none (pure v15)'} | "
          f"sweep={CFG.SWEEP_MODE} | compensate_steps={CFG.COMPENSATE_STEPS}")
    print(f"[preset] v18 flags ON: {on18 or 'none (v17 behaviour)'}")
    on182 = [f for f in _V182_FLAGS if getattr(CFG, f)]
    print(f"[preset] v18.2 flags ON: {on182 or 'none'} | curve_mode={CFG.CURVE_MODE}")
    on183 = [f for f in _V183_FLAGS if getattr(CFG, f)]
    print(f"[preset] v18.3 flags ON: {on183 or 'none'} | apgd_cond1={CFG.APGD_COND1_MODE}")
    print("[preset] loss terms: ce=%.2f%s margin=%.2f contrast=%.2f evasion=%.2f "
          "freerun=%.2f preserve=%s"
          % (CFG.CE_WEIGHT, " (non-pivot only)" if CFG.CE_NONPIVOT_ONLY else "",
             CFG.MARGIN_WEIGHT, CFG.CONTRAST_WEIGHT, CFG.EVASION_WEIGHT,
             CFG.FREERUN_WEIGHT,
             ("lagrangian C=%.2f" % CFG.PRESERVE_C) if (CFG.PRESERVE_LOSS and
              CFG.PRESERVE_LAGRANGIAN) else
             ("fixed w=%.2f" % CFG.PRESERVE_WEIGHT) if CFG.PRESERVE_LOSS else "off"))
    if CFG.CURVE_MODE != "none":
        vals = (CFG.EPS_CURVE_VALUES if CFG.CURVE_MODE == "eps"
                else CFG.MASK_CURVE_VALUES)
        print(f"[preset] CURVE MODE: {len(vals)} points per example -> runtime is "
              f"{len(vals)}x. Every point is recorded; NOTHING is selected as best.")
    if len(on) > 1:
        print("[preset] WARNING: more than one v16 flag is on. v16 changed four "
              "things at once and the -0.190 regression could not be attributed "
              "to any of them. Prefer PRESET='ablate_<one>'.")


_apply_preset()


def n_train_variants():
    """Number of prompt variants build_train_bank will emit. PROMPT_EOT samples
    ONE per step, so this divides the optimisation budget.

    Under PARAPHRASE_POOL the notion no longer applies: every step draws a fixed
    stratified subset of DRO_SUBSET probes from the whole pool, so adding
    templates does NOT dilute anyone's budget. Returning 3 keeps
    compensated_steps() a no-op in that regime, which is what we want."""
    if CFG.PARAPHRASE_POOL:
        return 3
    return 3 + (1 if CFG.ADD_APPEAR_FRAME else 0)


def compensated_steps(n_steps):
    """Keep per-variant step count constant when a variant is added, so an
    ablation measures the variant and not the budget cut."""
    if not getattr(CFG, "COMPENSATE_STEPS", False):
        return n_steps
    return int(round(n_steps * n_train_variants() / 3.0))


def vlog(msg, level=1):
    if CFG.VERBOSE and level <= CFG.VERBOSITY:
        print(msg)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("[WARN] No CUDA GPU — very slow / may not fit.")
else:
    # =====================================================================
    # v18.7 IDEA-B: DETERMINISM. This MUST live in the attack process.
    #
    # run_arm.py launches every arm with subprocess.call([sys.executable,
    # path]) -- a FRESH interpreter. torch.use_deterministic_algorithms() and
    # torch.backends.cudnn.deterministic set in a Colab notebook therefore do
    # NOT reach the attack at all. Only environment variables cross the
    # boundary. Setting determinism in the notebook and then running an arm
    # measures nothing, which is exactly what happened to pair_e/pair_f.
    #
    # Worse, the line this replaces set cudnn.benchmark = True. Benchmark mode
    # TIMES several convolution algorithms at runtime and keeps the fastest --
    # so the algorithm chosen depends on transient machine load, and two runs
    # of identical code can take different numerical paths. It is a documented
    # source of run-to-run nondeterminism and it was on for every run in this
    # project. Pinning the random streams while leaving this on is like
    # sealing the windows and leaving the door open.
    #
    # TF32 is also disabled under DETERMINISTIC: it truncates matmul mantissas
    # to 10 bits, which does not by itself cause nondeterminism but does
    # amplify whatever remains, and 60 steps of sign-gradient APGD is a very
    # efficient amplifier.
    torch.backends.cuda.matmul.allow_tf32 = not CFG.DETERMINISTIC
    torch.backends.cudnn.allow_tf32 = not CFG.DETERMINISTIC
    torch.backends.cudnn.benchmark = not CFG.DETERMINISTIC
    if CFG.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            # warn_only=True: some LLaVA ops have no deterministic kernel. We
            # want the run to proceed AND to name them, because an unnamed
            # silent fallback is how a determinism test passes without ever
            # having been performed.
            torch.use_deterministic_algorithms(True, warn_only=True)
            print("[init] DETERMINISTIC=True | cudnn.benchmark=False "
                  "tf32=False use_deterministic_algorithms=True(warn_only)")
            print("[init] any 'does not have a deterministic implementation' "
                  "warning below names an op that is STILL random -- it is the "
                  "reason two identical runs can differ.")
        except Exception as e:                                  # noqa: BLE001
            print(f"[init][WARN] use_deterministic_algorithms failed: {e}")
    else:
        print("[init] DETERMINISTIC=False | cudnn.benchmark=True tf32=True "
              "-- run-to-run results WILL differ even with CRN on.")
    vlog(f"[init] CUDA: {torch.cuda.get_device_name(0)}", 1)
torch.manual_seed(CFG.SEED); random.seed(CFG.SEED)


# =============================================================================
# 1b. Checkpoint / resume
# =============================================================================
def _ckpt_dir():
    d = os.path.join(CFG.OUTPUT_DIR, "checkpoints"); os.makedirs(d, exist_ok=True); return d

def _atomic_write_json(path, obj):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _atomic_save_pil(pil_img, path):
    tmp = f"{path}.tmp.{os.getpid()}.png"; pil_img.save(tmp); os.replace(tmp, path)

def _read_json(path):
    with open(path) as f: return json.load(f)

def example_key(img_name, subj, obj, rel_true):
    raw = f"{img_name}|{subj}|{obj}|{rel_true}".strip().lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

def _ex_done_path(key): return os.path.join(_ckpt_dir(), f"ex_{key}.done.json")
def _ex_skip_path(key): return os.path.join(_ckpt_dir(), f"ex_{key}.skip.json")

def set_example_seed(key, salt=0):
    """salt=0 reproduces v15 exactly. A non-zero salt gives an independent random
    init / EOT dither / prompt-EOT draw for the same example -- that is what makes
    the v16 restart sweep a genuine multi-start attack rather than three copies."""
    seed = (int(key[:8], 16) ^ (CFG.SEED & 0xffffffff)
            ^ (int(salt) * 0x9E3779B1)) & 0x7fffffff
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if CFG.CRN_ENABLE:
        crn_begin_run(key, salt)


# =============================================================================
# 1b-2. v18.7 IDEA-B: COMMON RANDOM NUMBERS.
#
#   Every stochastic draw becomes a pure function of
#       (example key, restart salt, phase, step, draw index)
#   instead of a function of RNG stream POSITION. Position depends on how many
#   times loss_and_grad is called, which depends on the arm (mint refreshes,
#   dual refreshes, CVaR emptying the probe set), so two arms on the same seed
#   currently ride different noise. Pinning the draws to indices makes arm A
#   and arm B see bit-identical dithers, inits and probe subsets, which is what
#   makes a paired delta measure the OBJECTIVE rather than the trajectory.
#
#   _CRN is module-level and explicitly reset per run, so a forgotten reset
#   shows up as an obviously wrong fingerprint rather than as silent drift.
# =============================================================================
_CRN = {"base": 0, "step": 0, "phase": "init", "on": False,
        "key": None, "salt": 0, "draws": 0, "fingerprints": []}
# Archive of the LAST COMPLETED ATTACK's stream, keyed by example.
#
# WHY THIS EXISTS. crn_begin_run() resets draws/fingerprints, and
# set_example_seed() -- which calls it -- is invoked a SECOND time after the
# attack, by the RUN_RANDOM_CONTROL block. So reading the live _CRN when the
# results record is built reports the random control's stream (0 draws, 0
# fingerprints), not the attack's. The mechanism was fine; the telemetry was
# being wiped one call before it was recorded. Archiving at the end of
# apgd_attack makes the report describe the attack no matter how many times
# the stream is re-seeded afterwards.
_CRN_ARCHIVE = {}


def crn_begin_run(key, salt=0):
    """Reset the CRN stream for one (example, restart) pair. Called from
    set_example_seed so it cannot be forgotten at a call site."""
    base = (int(str(key)[:8], 16)
            ^ (int(CFG.CRN_STREAM_SEED) & 0xffffffff)
            ^ (int(salt) * 0x9E3779B1)) & 0x7fffffff
    _CRN.update({"base": base, "step": 0, "phase": "init", "on": bool(CFG.CRN_ENABLE),
                 "key": key, "salt": int(salt), "draws": 0, "fingerprints": []})
    if CFG.CRN_DEBUG:
        print(f"    [crn] stream reset  key={key} salt={salt} base=0x{base:08x} "
              f"strict_probes={CFG.CRN_STRICT_PROBES}")


def crn_set_step(k, phase="main"):
    """Called by apgd_attack every iteration. `phase` separates the main
    gradient call from the extra refresh calls, so a refresh that fires in one
    arm and not another cannot shift the main stream."""
    _CRN["step"] = int(k); _CRN["phase"] = str(phase)


def _stable_hash(s):
    """CRC32 of the UTF-8 bytes.

    NOT python's builtin hash(). CPython salts str/bytes hashing with a
    per-PROCESS random seed (PEP 456) unless PYTHONHASHSEED is pinned, so
    hash("dither") returns a different value in every interpreter. Using it
    inside the CRN seed made the "pinned" streams reproducible only WITHIN one
    process -- every fresh `python run_arm.py` drew different dithers and a
    different random init, which is the exact failure CRN exists to prevent,
    hidden behind a flag that said it was fixed. crc32 is stable across
    processes, machines and python versions."""
    return zlib.crc32(str(s).encode("utf-8")) & 0xffffffff


def _crn_seed(tag, idx=0):
    """Deterministic 63-bit seed for one specific draw. Process-independent."""
    h = (_CRN["base"] * 0x27220A95) & 0xffffffffffffffff
    h ^= _stable_hash(tag) * 0x9E3779B97F4A7C15
    h ^= (int(_CRN["step"]) + 1) * 0xBF58476D1CE4E5B9
    h ^= (_stable_hash(_CRN["phase"]) & 0xffff) * 0x94D049BB133111EB
    h ^= (int(idx) + 1) * 0xD6E8FEB86659FD93
    return h & 0x7fffffffffffffff


def crn_generator(tag, idx=0, dev=None):
    """A fresh torch.Generator pinned to (run, phase, step, tag, idx)."""
    g = torch.Generator(device=(dev or "cpu"))
    g.manual_seed(_crn_seed(tag, idx))
    return g


def crn_rand_like(x, tag, idx=0):
    """Uniform[0,1) tensor shaped like x, drawn from the pinned stream.

    Generated on CPU then moved, because CUDA generator streams are not
    guaranteed bit-identical across devices/driver versions -- and the whole
    point of CRN is bit-identity across runs."""
    if not _CRN["on"]:
        return torch.rand_like(x)
    _CRN["draws"] += 1
    g = crn_generator(tag, idx, dev="cpu")
    r = torch.rand(x.shape, generator=g, dtype=torch.float32, device="cpu")
    return r.to(device=x.device, dtype=x.dtype)


def crn_random(tag, idx=0):
    """A python Random pinned to the same coordinates (for probe subsets)."""
    return random.Random(_crn_seed(tag, idx))


def crn_fingerprint(name, t):
    """Cheap scalar summary of a drawn tensor. Two arms that shared the stream
    print IDENTICAL fingerprints; that is the proof, not the intention."""
    if not (CFG.CRN_ENABLE and CFG.CRN_DEBUG):
        return
    if _CRN["step"] not in tuple(CFG.CRN_FINGERPRINT_STEPS):
        return
    with torch.no_grad():
        tf = t.detach().float()
        fp = (float(tf.sum()), float(tf.abs().max()), float(tf.flatten()[:64].sum()))
    _CRN["fingerprints"].append({"step": _CRN["step"], "phase": _CRN["phase"],
                                 "name": name, "fp": [round(v, 8) for v in fp]})
    print(f"    [crn] fp step={_CRN['step']:<4d} phase={_CRN['phase']:<7s} "
          f"{name:<12s} sum={fp[0]:+.6f} max={fp[1]:.6f} head={fp[2]:+.6f}")


def crn_archive():
    """Snapshot the current stream as the completed attack's record. Called at
    the end of apgd_attack, before anything else can re-seed."""
    _CRN_ARCHIVE[(_CRN.get("key"), _CRN.get("salt", 0))] = {
        "base": int(_CRN["base"]), "draws": int(_CRN["draws"]),
        "fingerprints": list(_CRN["fingerprints"])}


def crn_report(key=None):
    """Everything needed to prove two arms were paired, for results.json.

    Reads the ARCHIVE, not the live stream: by the time the results record is
    built, RUN_RANDOM_CONTROL has already re-seeded and zeroed the live
    counters. Falls back to the live stream only when nothing was archived,
    and flags that case rather than reporting an empty record as if it were a
    real measurement."""
    k = key if key is not None else _CRN.get("key")
    best, src = None, "live"
    for (ak, asalt), v in _CRN_ARCHIVE.items():
        if ak != k:
            continue
        if best is None or v["draws"] > best["draws"]:
            best, src = v, "archive"
    if best is None:
        best = {"base": int(_CRN["base"]), "draws": int(_CRN["draws"]),
                "fingerprints": list(_CRN["fingerprints"])}
    out = {"enable": bool(CFG.CRN_ENABLE),
           "strict_probes": bool(CFG.CRN_STRICT_PROBES),
           "stream_seed": int(CFG.CRN_STREAM_SEED),
           "source": src, **best}
    if CFG.CRN_ENABLE and out["draws"] == 0:
        # A CRN-enabled run that consumed zero pinned draws is NOT paired,
        # whatever the flag says. Say so in the record instead of letting the
        # audit pass on an empty fingerprint set.
        out["WARNING"] = ("CRN enabled but 0 pinned draws recorded -- this arm "
                          "is NOT verifiably paired. Do not report paired "
                          "deltas against it.")
    return out


# =============================================================================
# 1c. Resilient I/O + best-effort Google-Drive mirror
# =============================================================================
def _io_retry(fn):
    last = None
    for i in range(getattr(CFG, "IO_RETRIES", 5)):
        try: return fn()
        except OSError as e:
            last = e; time.sleep(1.5 * (i + 1))
    print(f"[io] gave up after {getattr(CFG, 'IO_RETRIES', 5)} retries: {last!r}")
    return None

def _safe_copy_tree(src, dst):
    if not os.path.isdir(src): return
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        if _io_retry(lambda out=out: (os.makedirs(out, exist_ok=True) or True)) is None:
            continue
        for fn in files:
            s = os.path.join(root, fn); d = os.path.join(out, fn)
            try:
                if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
                    continue
            except OSError:
                pass
            _io_retry(lambda s=s, d=d: shutil.copyfile(s, d))

def sync_to_drive():
    if not getattr(CFG, "DRIVE_MIRROR_DIR", None): return
    _safe_copy_tree(CFG.OUTPUT_DIR, CFG.DRIVE_MIRROR_DIR)

def restore_from_drive():
    if not getattr(CFG, "DRIVE_MIRROR_DIR", None): return
    if os.path.isdir(CFG.DRIVE_MIRROR_DIR):
        _safe_copy_tree(CFG.DRIVE_MIRROR_DIR, CFG.OUTPUT_DIR)
        print("[mirror] restored prior outputs from Drive (for resume).")


# =============================================================================
# 2. Spatial-relation knowledge
# =============================================================================
OPPOSITE = {
    "to the left of": "to the right of", "to the right of": "to the left of",
    "left of": "right of", "right of": "left of",
    "at the left side of": "at the right side of",
    "at the right side of": "at the left side of",
    "above": "below", "below": "above", "over": "under", "under": "over",
    "on top of": "beneath", "beneath": "on top of",
    "on": "under",                                   # <- What'sUp controlled_images
    "in front of": "behind", "behind": "in front of", "ahead of": "behind",
    "inside": "outside", "outside": "inside",
    "near": "far from", "far from": "near",
}
DIRECTION_KEYWORDS = {
    "right": ["right"], "left": ["left"],
    "above": ["above", "over ", "on top", "top of", "higher", " on the "],
    "below": ["below", "under", "beneath", "bottom", "lower"],
    "front": ["front", "ahead"], "behind": ["behind", "back of", "rear"],
    "inside": ["inside", "within"], "outside": ["outside"],
    "near": ["near", "close", "next to", "beside", "adjacent"],
    "far": ["far", "distant", "away from", "at a distance", "separated from", "apart from", "across from"],
}

PIVOT_WORDS = {
    "right":   ["right"],
    "left":    ["left"],
    "above":   ["above", "higher", "top", "over"],
    "below":   ["below", "lower", "beneath", "under", "bottom"],
    "front":   ["front", "ahead"],
    "behind":  ["behind", "rear", "back"],
    "inside":  ["inside", "within"],
    "outside": ["outside"],
    "near":    ["near", "close", "beside"],
    "far":     ["far", "distant", "away"],
}

def relation_group(rel):
    r = rel.lower().strip()
    if "right" in r:  return "right"
    if "left" in r:   return "left"
    if "front" in r or "ahead" in r: return "front"
    if "behind" in r or "back" in r: return "behind"
    if r == "on" or "above" in r or "over" in r or "on top" in r or "top" in r: return "above"
    if "below" in r or "under" in r or "beneath" in r or "bottom" in r: return "below"
    if "inside" in r or "within" in r: return "inside"
    if "outside" in r: return "outside"
    if "far" in r: return "far"
    if "near" in r or "next to" in r or "close" in r: return "near"
    return None

def expresses_relation(answer, rel):
    """v18.0 scorer: bare substring match over the whole answer. Kept for the
    SCORER_STRICT=False ablation and used internally by the strict scorer."""
    g = relation_group(rel)
    if g is None: return rel.lower() in answer.lower()
    a = answer.lower()
    return any(kw in a for kw in DIRECTION_KEYWORDS[g])


# --- v18.1 strict scorer -----------------------------------------------------
# Clause splitter. Subordinators matter more than punctuation here: the observed
# false negative was "..., with the sheep following behind it", where the true
# relation sits in a clause whose subject is the OBJECT.
_CLAUSE_RE = _re.compile(
    r",\s+|;\s+|\.\s+|\s+(?:while|whereas|although|though|but|and|with|as|since)\s+")

# Negation cues. Checked only to the LEFT of the matched relation word inside the
# same clause, so "the bowl is to the right, not the left" still scores as a hit.
_NEG_CUES = ("not ", "n't", "isn't", "aren't", "wasn't", "weren't", "cannot",
             "can not", "never", "neither", "nor ", "no longer", "rather than",
             "instead of", "opposite of", "incorrect", "false")

# Answers that deny the scene rather than relocate the object. On a "no" probe
# these score as hits under any relation scorer, which is precisely how a
# perturbation that has destroyed the image can look like a successful lie.
_DEGEN_CUES = ("does not show", "doesn't show", "does not contain",
               "doesn't contain", "there is no", "there are no", "not visible",
               "cannot see", "can't see", "no such", "is not present",
               "are not present", "does not appear in", "not depicted")


def _split_clauses(answer):
    parts = [c.strip() for c in _CLAUSE_RE.split(answer.lower()) if c and c.strip()]
    return parts or [answer.strip().lower()]


def _mentions(clause, phrase):
    """Head-noun match. Multi-word entities ('potted plant') count if any content
    token appears, so article/adjective drift does not break the scoping."""
    toks = [t for t in _norm_tokens(phrase) if len(t) > 2]
    if not toks:
        return phrase.strip().lower() in clause
    return any(t in clause for t in toks)


def _rel_hit_index(clause, rel):
    """Earliest index at which `rel`'s keyword family appears, or -1."""
    g = relation_group(rel)
    kws = DIRECTION_KEYWORDS[g] if g is not None else [rel.lower()]
    idxs = [clause.find(k) for k in kws]
    idxs = [i for i in idxs if i >= 0]
    return min(idxs) if idxs else -1


def _negated_before(clause, idx):
    if idx < 0:
        return False
    left = clause[:idx]
    return any(cue in left for cue in _NEG_CUES)


def is_degenerate(answer, subj=None, obj=None):
    """True when the answer denies the scene instead of stating a relation."""
    if not CFG.DEGENERATE_CHECK or not answer:
        return False
    a = answer.lower()
    if any(cue in a for cue in _DEGEN_CUES):
        return True
    if subj and _re.search(r"\bno\s+" + _re.escape(str(subj).lower().split()[-1]), a):
        return True
    return False


def score_open(answer, rt, rg, subj=None, obj=None):
    """Returns (hit, reason). Strict scorer for open-ended probes.

    Rules, in order:
      1. The TARGET relation must appear, un-negated, in the clause that mentions
         the subject (falling back to the first clause when the subject is not
         named, which is normal for answers like "In front.").
      2. The TRUE relation disqualifies the answer only if it appears in that
         SAME clause. A true-relation word in a clause about the object is
         describing the object's converse position and is not a contradiction.
      3. A degenerate answer is never a hit."""
    if not answer:
        return False, "empty"
    if not CFG.SCORER_STRICT:
        ok = expresses_relation(answer, rg) and not expresses_relation(answer, rt)
        return ok, "v18.0-substring"
    if is_degenerate(answer, subj, obj):
        return False, "degenerate"

    clauses = _split_clauses(answer)
    if CFG.SCORER_CLAUSE and subj:
        primary = next((c for c in clauses if _mentions(c, subj)), clauses[0])
    else:
        primary = clauses[0] if CFG.SCORER_CLAUSE else answer.lower()

    i_tgt = _rel_hit_index(primary, rg)
    if i_tgt < 0:
        # target may sit in a later clause of a multi-clause answer
        for c in clauses:
            if _rel_hit_index(c, rg) >= 0 and not (
                    CFG.SCORER_CLAUSE and subj and not _mentions(c, subj)):
                primary, i_tgt = c, _rel_hit_index(c, rg)
                break
    if i_tgt < 0:
        return False, "target_absent"
    if CFG.SCORER_NEGATION and _negated_before(primary, i_tgt):
        return False, "target_negated"
    i_true = _rel_hit_index(primary, rt)
    if i_true >= 0 and not (CFG.SCORER_NEGATION and _negated_before(primary, i_true)):
        return False, "true_relation_in_subject_clause"
    return True, "ok"


def yesno(answer):
    """Word-boundary matched. The old prefix test fired on 'Nothing', 'None'
    and 'Northwest' because it only checked startswith('no')."""
    a = answer.strip().lower()
    head = a[:16]
    if _re.search(r"\byes\b", head): return "yes"
    if _re.search(r"\bno\b|\bnope\b|\bincorrect\b", head): return "no"
    return None

def relation_family(rel):
    g = relation_group(rel)
    if g in ("left", "right"):     return "lateral"
    if g in ("above", "below"):    return "vertical"
    if g in ("front", "behind"):   return "depth"
    if g in ("inside", "outside"): return "containment"
    if g in ("near", "far"):       return "proximity"
    return "other"

def _norm_tokens(s):
    s = _re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return [t for t in s.split() if t]

def token_f1(a, b):
    from collections import Counter
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    common = Counter(ta) & Counter(tb); n = sum(common.values())
    if n == 0: return 0.0
    prec, rec = n / len(ta), n / len(tb)
    return 2 * prec * rec / (prec + rec)


# =============================================================================
# 3. Probes (consistency set + paraphrase bank) + scoring
# =============================================================================
def forced_choice_question(subj, obj, rt):
    fam = relation_family(rt)
    if fam == "lateral":      return f"Is the {subj} on the left or the right of the {obj}?"
    if fam == "vertical":     return f"Is the {subj} higher or lower than the {obj}?"
    if fam == "depth":        return f"From the camera's view, is the {subj} in front of or behind the {obj}?"
    if fam == "containment":  return f"Is the {subj} inside or outside the {obj}?"
    if fam == "proximity":    return f"Is the {subj} near or far from the {obj}?"
    return f"Where is the {subj} compared to the {obj}?"

def _group_choice_words(rg):
    g = relation_group(rg)
    table = {
        "right":   ["right"],
        "left":    ["left"],
        "above":   ["above", "higher", "on top"],
        "below":   ["below", "lower", "beneath", "under"],
        "front":   ["in front", "ahead"],
        "behind":  ["behind"],
        "inside":  ["inside", "within"],
        "outside": ["outside"],
        "near":    ["near", "close"],
        "far":     ["far"],
    }
    return table.get(g, [rg])

def build_target_answers(subj, obj, rt, rg, max_refs=None):
    if max_refs is None: max_refs = CFG.MAX_OPEN_REFS
    if not CFG.MULTI_REF:
        return [f"The {subj} is {rg} the {obj}."]
    words = _group_choice_words(rg)
    cands = [f"The {subj} is {rg} the {obj}."]
    cands += [f"{w[0].upper() + w[1:]}." for w in words]
    cands += [f"It is {rg} the {obj}.", f"The {subj} appears {rg} the {obj}."]
    good, seen = [], set()
    for c in cands:
        k = c.strip().lower()
        if k in seen: continue
        seen.add(k)
        if score_lie(c, "open", rt, rg):
            good.append(c)
        if len(good) >= max_refs: break
    return good or [f"The {subj} is {rg} the {obj}."]

def yesno_refs(polarity):
    """OPT-G: give soft-min more than one surface form to aim at. The eval scorer
    (yesno()) only inspects the prefix, so every form here scores identically."""
    if polarity == "yes":
        cands = ["Yes.", "Yes, it is.", "Yes, that is correct."]
    else:
        cands = ["No.", "No, it is not.", "No, that is incorrect."]
    if not CFG.MULTI_REF_YESNO:
        return cands[:1]
    return cands[:max(1, CFG.MAX_YESNO_REFS)]


# =============================================================================
# 3b. IDEA-3: paraphrase pool + deterministic disjoint TRAIN / EVAL split
# =============================================================================
# {s}=subject {o}=object {r}=relation phrase. Every template must be answerable
# by the SAME target answers, so the scorer stays valid across the whole pool.
OPEN_TEMPLATES = [
    "Describe the position of the {s} relative to the {o}.",
    "Where is the {s} in this image relative to the {o}?",
    "In this photo, where is the {s} with respect to the {o}?",
    "Relative to the {o}, how is the {s} positioned?",
    "Tell me the location of the {s} compared to the {o}.",
    "How is the {s} placed in relation to the {o}?",
    "In one short phrase, where is the {s} compared to the {o}?",
    "Where does the {s} appear relative to the {o}?",
    "From the camera's viewpoint, describe where the {s} is relative to the {o}.",
    "State the spatial relation between the {s} and the {o}.",
    "Looking at this image, how would you describe the {s}'s position versus the {o}?",
    "Briefly, what is the position of the {s} with respect to the {o}?",
    "Can you say where the {s} sits in relation to the {o}?",
    "Report the location of the {s}, using the {o} as the reference point.",
    "With the {o} as the reference, where is the {s}?",
    "Using one spatial term, relate the {s} to the {o}.",
    "How does the {s} stand relative to the {o} in this picture?",
    "Please indicate the placement of the {s} compared to the {o}.",
    "What is the position of the {s} when compared with the {o}?",
    "Describe, in spatial terms, the {s} with respect to the {o}.",
]

# --- v18.4: TRAIN-ONLY open templates ---------------------------------------
# Widen open coverage in TRAIN to close the held-out open-probe generalisation
# gap. These are the augmentation route recommended alongside uniform CVaR.
#
# Two invariants this respects, both load-bearing:
#   1. They NEVER enter the held-out eval. _pool_split() below computes the
#      eval/train split over the ORIGINAL _N_OPEN_BASE templates only, then
#      appends these extra indices to TRAIN. So the frozen eval set is byte-
#      identical to the v18.3 run (results__57_) and held-out SBC stays
#      comparable -- the leak-free guarantee is untouched.
#   2. They must be answerable by the SAME target answers as every other
#      template (build_target_answers), so the scorer stays valid.
# They deliberately push the model toward committing to ONE spatial phrase,
# which is exactly what the "next to"/"near"/"under"/"right of" dodges avoid.
OPEN_TEMPLATES_TRAIN_EXTRA = [
    "In one spatial phrase only, the {s} is where in relation to the {o}?",
    "Give the single spatial relation of the {s} to the {o}.",
    "Fill in the blank with a direction: the {s} is ___ the {o}.",
    "Name the one relation that locates the {s} relative to the {o}.",
    "Answer with a single spatial phrase: the {s} is positioned how versus the {o}?",
    "State only the spatial relation of the {s} with respect to the {o}.",
    "Write a brief description of this image that specifies where the {s} is in relation to the {o}.",
    "Caption this photo in one sentence by stating the position of the {s} relative to the {o}.",
    "Where can we see the {s}? Give its spatial position relative to the {o}.",
    "We can see the {o} in this picture. ASSISTANT: Yes, the {o} is present. USER: Where is the {s} positioned relative to it?",
]

# Number of BASE (splittable) open templates. Captured BEFORE the extras are
# appended so the eval split is computed over exactly the original set.
_N_OPEN_BASE = len(OPEN_TEMPLATES)
OPEN_TEMPLATES = OPEN_TEMPLATES + OPEN_TEMPLATES_TRAIN_EXTRA

# Yes/no FRAMES. The same frame is used for both polarities: inserting the
# TARGET relation makes it a "yes" probe, the TRUE relation a "no" probe. The
# v17 diagnosis was a frame gap ("appear" was never trained), so frames — not
# just relations — are what has to be split.
YESNO_FRAMES = [
    "Is the {s} {r} the {o}? Answer yes or no.",
    "Would you say the {s} is {r} the {o}? yes or no.",
    "Is it correct that the {s} is {r} the {o}? yes or no.",
    "Does the {s} appear {r} the {o}? Answer yes or no.",
    "Answer yes or no: the {s} is {r} the {o}.",
    "In this image, is the {s} {r} the {o}? yes or no.",
    "Can you confirm that the {s} is {r} the {o}? Answer yes or no.",
    "Do you agree that the {s} is {r} the {o}? yes or no.",
    "From what you see, is the {s} {r} the {o}? Answer yes or no.",
    "Is the statement \"the {s} is {r} the {o}\" correct? yes or no.",
    "Judging by the photo, is the {s} {r} the {o}? yes or no.",
    "Would it be accurate to say the {s} is {r} the {o}? yes or no.",
    "Please answer yes or no: is the {s} {r} the {o}?",
    "Based on the image, is the {s} {r} the {o}? Answer yes or no.",
]


def _pool_split():
    """Deterministic, example-independent split. Uses its OWN Random instance so
    it can never be perturbed by set_example_seed() or by the EOT draws."""
    rnd = random.Random(CFG.POOL_SPLIT_SEED)
    # v18.4: split over the BASE templates only (range(_N_OPEN_BASE)). Because
    # the seed and the list length are unchanged from v18.3, this shuffle yields
    # the identical permutation, so open_eval is byte-identical to results__57_.
    op = list(range(_N_OPEN_BASE)); rnd.shuffle(op)
    yn = list(range(len(YESNO_FRAMES)));   rnd.shuffle(yn)
    if CFG.POOL_RECLAIM_UNUSED:
        # v18.8 (0). build_eval_probes consumes only the FIRST EVAL_N_* indices;
        # everything after them was previously in neither set. Cutting the split
        # exactly at EVAL_N_* sends those indices to TRAIN and leaves the eval
        # set byte-identical, because the shuffle (and therefore the ordering of
        # the first EVAL_N_* entries) is unchanged.
        n_o = int(CFG.EVAL_N_OPEN)
        n_y = int(CFG.EVAL_N_YESNO)
    else:
        n_o = max(CFG.EVAL_N_OPEN,  int(round(len(op) * CFG.POOL_EVAL_FRAC)))
        n_y = max(CFG.EVAL_N_YESNO, int(round(len(yn) * CFG.POOL_EVAL_FRAC)))
    # v18.4: TRAIN-ONLY extras appended AFTER the split -> they can never be eval.
    extra_train = list(range(_N_OPEN_BASE, len(OPEN_TEMPLATES)))
    split = {"open_eval":  op[:n_o],  "open_train":  sorted(op[n_o:]) + extra_train,
             "yn_eval":    yn[:n_y],  "yn_train":    sorted(yn[n_y:])}
    # Hard guarantee: no template index is ever on both sides.
    assert not (set(split["open_eval"]) & set(split["open_train"]))
    assert not (set(split["yn_eval"])   & set(split["yn_train"]))
    return split


POOL = _pool_split()


def _assert_eval_split_stable():
    """v18.8 (0). POOL_RECLAIM_UNUSED must move templates into TRAIN and must
    NOT move the eval set, or every held-out SBC in this project stops being
    comparable with every earlier run. Verify it instead of asserting it in a
    comment: build the split both ways and compare the indices that
    build_eval_probes actually consumes.

    This runs at import time and is cheap. If it ever fires, do not 'fix' it by
    relaxing the check -- the reclaim changed the measurement."""
    was = bool(CFG.POOL_RECLAIM_UNUSED)
    try:
        CFG.POOL_RECLAIM_UNUSED = False
        a = _pool_split()
        CFG.POOL_RECLAIM_UNUSED = True
        b = _pool_split()
    finally:
        CFG.POOL_RECLAIM_UNUSED = was
    no, ny = int(CFG.EVAL_N_OPEN), int(CFG.EVAL_N_YESNO)
    assert a["open_eval"][:no] == b["open_eval"][:no], (
        "POOL_RECLAIM_UNUSED moved the OPEN eval templates: %s -> %s"
        % (a["open_eval"][:no], b["open_eval"][:no]))
    assert a["yn_eval"][:ny] == b["yn_eval"][:ny], (
        "POOL_RECLAIM_UNUSED moved the YES/NO eval frames: %s -> %s"
        % (a["yn_eval"][:ny], b["yn_eval"][:ny]))
    n_recl = ((len(b["open_train"]) - len(a["open_train"]))
              + 2 * (len(b["yn_train"]) - len(a["yn_train"])))
    return {"open_eval": a["open_eval"][:no], "yn_eval": a["yn_eval"][:ny],
            "reclaimed_probes": n_recl,
            "train_probes_off": len(a["open_train"]) + 2 * len(a["yn_train"]),
            "train_probes_on":  len(b["open_train"]) + 2 * len(b["yn_train"])}


_EVAL_SPLIT_AUDIT = _assert_eval_split_stable()


def describe_pool_split():
    ev_o = POOL["open_eval"][:CFG.EVAL_N_OPEN]
    ev_y = POOL["yn_eval"][:CFG.EVAL_N_YESNO]
    n_spec = 4 if getattr(CFG, "EVAL_EXPANDED_PROBES", True) else 0
    n_tot_eval = len(ev_o) + n_spec + 2 * len(ev_y)
    print("[pool] TRAIN open=%d  yes/no frames=%d   ->  %d train probes/example"
          % (len(POOL["open_train"]), len(POOL["yn_train"]),
             len(POOL["open_train"]) + 2 * len(POOL["yn_train"])
             + (1 if CFG.ADD_FORCED_CHOICE else 0)))
    print("[pool] EVAL  open idx=%s (+%d special)  frame idx=%s  -> %d held-out probes/example"
          % (ev_o, n_spec, ev_y, n_tot_eval))
    print("[pool] held-out open templates:")
    for i in ev_o:
        print("         - [o%d] %s" % (i, OPEN_TEMPLATES[i]))
    if n_spec:
        print("         - [mc_eval]  Multiple choice: relative to the {o}, is the {s} (A) ... or (B) ...?")
        print("         - [wh_eval]  Where is the {s}? State its position relative to the {o}.")
        print("         - [cap_eval] Write a one-sentence scene caption stating where the {s} is relative to the {o}.")
        print("         - [fup_eval] 2-turn follow-up: Turn 1 locates {o}, Turn 2 asks where {s} is relative to it.")
    print("[pool] held-out yes/no frames:")
    for i in ev_y:
        print("         - [y%d/n%d] %s" % (i, i, YESNO_FRAMES[i]))
    print("[pool] NOTE: no eval template is ever optimised (train/eval overlap = 0).")



def converse_relations(rt, rg):
    """Idea N2 (Bidirectional Converse Cycle-Consistency):
    Returns (rt_converse, rg_converse) for the swapped entity pair (obj, subj).
      - Anti-symmetric families (lateral, vertical, depth):
          s R_target o  <=>  o R_true s
          (e.g., if cat is 'left of' laptop, laptop is 'right of' cat).
          So for the converse query about (obj, subj), the LIE target is `rt`
          and the TRUE relation to suppress is `rg`!
      - Symmetric family (proximity: 'near' / 'far from'):
          s R_target o  <=>  o R_target s
          (if cat is 'far from' couch, couch is 'far from' cat).
          So for (obj, subj), the LIE target is still `rg` and truth is `rt`.
      - Containment ('inside' / 'outside'):
          If s is 'outside' o (target='outside'), o is also 'outside' s (or not containing s);
          only inject converse open/yn on symmetric/anti-symmetric families where the
          converse preposition is exact."""
    fam = relation_family(rt)
    if fam in ("lateral", "vertical", "depth"):
        return rg, rt   # (converse_true = rg, converse_target = rt)
    if fam == "proximity":
        return rt, rg   # symmetric: (converse_true = rt, converse_target = rg)
    return None, None


def build_train_pool(subj, obj, rt, rg):
    """Flat list of {q, answers, kind, pkey}. Replaces the variant-of-variants
    structure so a step can draw an arbitrary stratified subset."""
    tgt = build_target_answers(subj, obj, rt, rg)

    if not CFG.PARAPHRASE_POOL:
        # v17 path: flatten the 3-variant bank, keep the variant tag so
        # _active_probes() can still do PROMPT_EOT variant sampling.
        pool = []
        for vi, var in enumerate(build_train_bank(subj, obj, rt, rg)):
            for pi, pr in enumerate(var):
                pool.append({**pr, "pkey": "v%dp%d" % (vi, pi), "variant": vi})
        return pool

    # Step 3 Baseline B1: Single-prompt targeted PGD (1 open question + 1 target answer)
    if str(getattr(CFG, "BASELINE_ARM", "full")) == "B1":
        i0 = POOL["open_train"][0] if POOL["open_train"] else 0
        return [{"q": OPEN_TEMPLATES[i0].format(s=subj, o=obj),
                 "answers": [f"The {subj} is {rg} the {obj}."],
                 "kind": "open", "pkey": "o%d" % i0, "variant": 0}]

    pool = []
    for i in POOL["open_train"]:
        pool.append({"q": OPEN_TEMPLATES[i].format(s=subj, o=obj),
                     "answers": tgt, "kind": "open", "pkey": "o%d" % i, "variant": 0})
    if CFG.ADD_FORCED_CHOICE:
        pool.append({"q": forced_choice_question(subj, obj, rt),
                     "answers": tgt, "kind": "open", "pkey": "fc", "variant": 0})
    for i in POOL["yn_train"]:
        f = YESNO_FRAMES[i]
        pool.append({"q": f.format(s=subj, o=obj, r=rg),
                     "answers": yesno_refs("yes"), "kind": "yes",
                     "pkey": "y%d" % i, "variant": 0})
        pool.append({"q": f.format(s=subj, o=obj, r=rt),
                     "answers": yesno_refs("no"), "kind": "no",
                     "pkey": "n%d" % i, "variant": 0})

    # ---- IDEA-N2: Bidirectional Converse Cycle-Consistency ------------------
    # Enforces Belief(s, rg, o) <=> Belief(o, rg_conv, s) on training templates
    # so the VLM edits its 3D spatial scene graph rather than memorizing a
    # one-way lexical prefix on `subj`. Zero overlap with held-out eval templates.
    if getattr(CFG, "NOVEL_CONVERSE_CYCLE", False) and str(getattr(CFG, "BASELINE_ARM", "full")) not in ("B0", "B1", "B2", "B3", "B4", "B5"):
        rt_conv, rg_conv = converse_relations(rt, rg)
        if rt_conv is not None and rg_conv is not None:
            tgt_conv = build_target_answers(obj, subj, rt_conv, rg_conv)
            n_co = max(0, int(getattr(CFG, "CONVERSE_OPEN_COUNT", 6)))
            for i in POOL["open_train"][:n_co]:
                pool.append({"q": OPEN_TEMPLATES[i].format(s=obj, o=subj),
                             "answers": tgt_conv, "kind": "open",
                             "pkey": "co%d" % i, "variant": 0,
                             "converse": True, "subj_row": obj, "obj_row": subj,
                             "rt_row": rt_conv, "rg_row": rg_conv})
            n_cy = max(0, int(getattr(CFG, "CONVERSE_YN_COUNT", 4)))
            for i in POOL["yn_train"][:n_cy]:
                f = YESNO_FRAMES[i]
                pool.append({"q": f.format(s=obj, o=subj, r=rg_conv),
                             "answers": yesno_refs("yes"), "kind": "yes",
                             "pkey": "cy%d" % i, "variant": 0,
                             "converse": True, "subj_row": obj, "obj_row": subj,
                             "rt_row": rt_conv, "rg_row": rg_conv})
                pool.append({"q": f.format(s=obj, o=subj, r=rt_conv),
                             "answers": yesno_refs("no"), "kind": "no",
                             "pkey": "cn%d" % i, "variant": 0,
                             "converse": True, "subj_row": obj, "obj_row": subj,
                             "rt_row": rt_conv, "rg_row": rg_conv})
    return pool


def build_train_bank(subj, obj, rt, rg):
    tgt = build_target_answers(subj, obj, rt, rg)
    fcq = forced_choice_question(subj, obj, rt)

    def O(q): return {"q": q, "answers": tgt, "kind": "open"}
    def Y(q): return {"q": q, "answers": yesno_refs("yes"), "kind": "yes"}
    def N(q): return {"q": q, "answers": yesno_refs("no"),  "kind": "no"}

    def variant(open_qs, y_q, n_q):
        probes = [O(q) for q in open_qs]
        if CFG.ADD_FORCED_CHOICE:
            probes.append(O(fcq))
        probes += [Y(y_q), N(n_q)]
        return probes

    bank = [
        variant([f"Describe the position of the {subj} relative to the {obj}.",
                 f"Where is the {subj} in this image relative to the {obj}?"],
                f"Is the {subj} {rg} the {obj}? Answer yes or no.",
                f"Is the {subj} {rt} the {obj}? Answer yes or no."),
        variant([f"In this photo, where is the {subj} with respect to the {obj}?",
                 f"Relative to the {obj}, how is the {subj} positioned?"],
                f"Would you say the {subj} is {rg} the {obj}? yes or no.",
                f"Would you say the {subj} is {rt} the {obj}? yes or no."),
        variant([f"Tell me the location of the {subj} compared to the {obj}.",
                 f"How is the {subj} placed in relation to the {obj}?"],
                # v16: suffix aligned to build_eval_probes ("yes or no." not
                # "yes/no."). Free; the frame already transferred at 20/20, this
                # just removes the last needless train/eval delta.
                f"Is it correct that the {subj} is {rg} the {obj}? yes or no.",
                f"Is it correct that the {subj} is {rt} the {obj}? yes or no."),
    ]

    # OPT-D: the frame build_eval_probes actually uses. Absent from v15, and it
    # accounts for 10 of the 23 failures on its own (see CFG note).
    if CFG.ADD_APPEAR_FRAME:
        bank.append(variant(
            [f"Where does the {subj} appear relative to the {obj}?",
             f"From the camera's viewpoint, describe where the {subj} is "
             f"relative to the {obj}."],
            f"Does the {subj} appear {rg} the {obj}? Answer yes or no.",
            f"Does the {subj} appear {rt} the {obj}? Answer yes or no."))
    return bank

PRESERVATION_QUESTIONS = [
    "What objects are in this image? List them.",
    "What is the main subject of this image?",
]

def heldout_mc_question(subj, obj, rt, rg):
    """Step 2 held-out binary choice probe (disjoint from train forced_choice_question).
    Avoids (A)/(B) letter labels so the model's answer states the spatial phrase
    directly rather than emitting 'A' or 'B' (which fails the strict text scorer)."""
    fam = relation_family(rt)
    if fam == "lateral":
        return f"Choose one option and state the relation: relative to the {obj}, is the {subj} to the left of it or to the right of it?"
    if fam == "vertical":
        return f"Choose one option and state the relation: relative to the {obj}, is the {subj} above it or below it?"
    if fam == "depth":
        return f"Choose one option and state the relation: relative to the {obj}, is the {subj} in front of it or behind it?"
    if fam == "containment":
        return f"Choose one option and state the relation: relative to the {obj}, is the {subj} inside it or outside it?"
    if fam == "proximity":
        return f"Choose one option and state the relation: relative to the {obj}, is the {subj} near it or far from it?"
    return f"Choose one option and state the relation: is the {subj} {rt} or {rg} the {obj}?"


def build_eval_probes(subj, obj, rt, rg):
    """Step 2 (v19): Expanded held-out evaluation probe suite (15 probes total,
    train/eval overlap = 0):
      - 5 held-out open paraphrases from POOL['open_eval'][:5] (o19, o13, o4, o9, o16)
      - 4 held-out structural open probes (when EVAL_EXPANDED_PROBES=True):
          * mc_eval  : held-out multiple-choice question
          * wh_eval  : direct 'Where is X?' question
          * cap_eval : scene caption describing spatial layout of X and Y
          * fup_eval : 2-turn conversational follow-up (Turn 1 locates {o}, Turn 2 asks {s})
      - 6 held-out yes/no probes from POOL['yn_eval'][:3] x {yes, no} (y6/n6, y11/n11, y5/n5)
    Also tags the original 7-probe subset (`LEGACY_7_PKEYS`) so `_sbc_and_coherence`
    reports both the 15-probe metrics (`SBC`, `coherence`) and the backward-compatible
    7-probe metrics (`SBC_7`, `coherence_7`)."""
    if not CFG.PARAPHRASE_POOL:
        axis_q = forced_choice_question(subj, obj, rt)
        return [
            (axis_q, "open", rg, "v17_axis"),
            (f"In one short phrase, where is the {subj} compared to the {obj}?", "open", rg, "v17_phrase"),
            (f"Relative to the {obj}, how is the {subj} positioned?", "open", rg, "v17_relative"),
            (f"Does the {subj} appear {rg} the {obj}? Answer yes or no.", "yes", None, "v17_appear"),
            (f"Is it correct that the {subj} is {rg} the {obj}? yes or no.", "yes", None, "v17_correct"),
            (f"Does the {subj} appear {rt} the {obj}? Answer yes or no.", "no", None, "v17_appear"),
            (f"Is it correct that the {subj} is {rt} the {obj}? yes or no.", "no", None, "v17_correct"),
        ]

    probes = []
    for i in POOL["open_eval"][:CFG.EVAL_N_OPEN]:
        probes.append((OPEN_TEMPLATES[i].format(s=subj, o=obj), "open", rg, "o%d" % i))

    if getattr(CFG, "EVAL_EXPANDED_PROBES", True):
        # 1) Held-out multiple-choice ("left or right?" / "(A) ... or (B) ...")
        probes.append((heldout_mc_question(subj, obj, rt, rg), "open", rg, "mc_eval"))
        # 2) Held-out direct "Where is X?" probe
        probes.append((f"Where is the {subj}? State its position relative to the {obj}.",
                       "open", rg, "wh_eval"))
        # 3) Held-out scene caption probe
        probes.append((f"Write a one-sentence scene caption stating where the {subj} is relative to the {obj}.",
                       "open", rg, "cap_eval"))
        # 4) Held-out 2-turn conversational follow-up probe
        #    Encoded as a 2-turn LLaVA dialogue so generate_batch runs it in a single pass
        probes.append((
            f"I see the {obj} in the scene. ASSISTANT: Yes, the {obj} is visible in the image. "
            f"USER: Follow-up question: where is the {subj} located relative to it?",
            "open", rg, "fup_eval"
        ))

    for i in POOL["yn_eval"][:CFG.EVAL_N_YESNO]:
        f = YESNO_FRAMES[i]
        probes.append((f.format(s=subj, o=obj, r=rg), "yes", None, "y%d" % i))
        probes.append((f.format(s=subj, o=obj, r=rt), "no", None, "n%d" % i))
    return probes


def assert_no_leakage(subj, obj, rt, rg):
    """Fail loudly rather than silently reporting an inflated SBC."""
    train_qs = {p["q"] for p in build_train_pool(subj, obj, rt, rg)}
    eval_qs  = {t[0] for t in build_eval_probes(subj, obj, rt, rg)}
    overlap  = train_qs & eval_qs
    if overlap and CFG.PARAPHRASE_POOL:
        raise RuntimeError("EVAL/TRAIN PROBE LEAKAGE: %r" % sorted(overlap))
    return overlap

def build_clean_check_probes(subj, obj, rt, rg):
    return (f"Is the {subj} {rt} the {obj}? Answer yes or no.",
            f"Is the {subj} {rg} the {obj}? Answer yes or no.")

def score_lie(answer, kind, rt, rg, subj=None, obj=None, with_reason=False):
    """subj/obj are optional so every v17 call site keeps working; pass them to
    get the clause-scoped behaviour. Yes/no probes get the degeneracy check too:
    'No. The image does not show a bed.' is a correct 'no' for the wrong reason
    and must not be silently banked as a successful lie."""
    if kind == "open":
        ok, why = score_open(answer, rt, rg, subj, obj)
    elif kind in ("yes", "no"):
        ok = (yesno(answer) == kind)
        why = "ok" if ok else "polarity"
        if ok and is_degenerate(answer, subj, obj):
            why = "degenerate"          # still counted; reported separately
    else:
        ok, why = False, "unknown_kind"
    return (ok, why) if with_reason else ok


# --- scorer regression suite -------------------------------------------------
# Every case below is a REAL answer taken from the v17 (n=23) or v18.0 (n=5) runs
# with the verdict the metric should have given. Run at startup; a change to the
# keyword tables or the clause splitter that breaks one of these is caught before
# it silently moves the headline number.
_SCORER_CASES = [
    # (answer, kind, rel_true, rel_target, subj, obj, expected_hit, note)
    ("The cow is walking in front of the sheep, with the sheep following behind it.",
     "open", "behind", "in front of", "cow", "sheep", True,
     "v18.0 FALSE NEGATIVE: true relation was in the object's clause"),
    ("The bed is located next to the potted plant.",
     "open", "right of", "left of", "bed", "potted plant", False,
     "genuine 'next to' evasion"),
    ("The bowl is not to the right of the banana.",
     "open", "at the left side of", "at the right side of", "bowl", "banana", False,
     "v18.0 FALSE POSITIVE: negation was ignored"),
    ("The bowl is to the right of the banana, not the left.",
     "open", "at the left side of", "at the right side of", "bowl", "banana", True,
     "negation after the target must not disqualify it"),
    ("The bowl is at the right side of the banana.",
     "open", "at the left side of", "at the right side of", "bowl", "banana", True,
     "plain hit"),
    ("The hot dog is on the right side of the plate, while the broccoli is on the left side.",
     "open", "at the right side of", "at the left side of", "hot dog", "plate", False,
     "subject clause carries the TRUE relation"),
    ("In front.",
     "open", "behind", "in front of", "cow", "sheep", True,
     "subject unnamed -> fall back to the first clause"),
    ("The car is positioned on top of the cake, with the cake serving as the base.",
     "open", "on top of", "beneath", "car", "cake", False,
     "target absent"),
    ("No. The image does not show a bed. It shows a person sitting on a chair.",
     "no", "right of", "left of", "bed", "potted plant", True,
     "counts for SBC but MUST be flagged degenerate"),
    ("Yes.", "no", "ahead of", "behind", "person", "cow", False, "wrong polarity"),
    ("Nothing is visible there.", "no", "ahead of", "behind", "person", "cow", False,
     "old yesno() matched 'no' inside 'Nothing'"),
    ("Yes, it would be accurate to say that the person is behind the cow.",
     "yes", "ahead of", "behind", "person", "cow", True, "plain yes"),
]

_DEGEN_CASES = [
    ("No. The image does not show a bed.", "bed", True),
    ("The bed is to the left of the potted plant.", "bed", False),
    ("There is no cow in this picture.", "cow", True),
]


def selftest_scorer(verbose=True):
    fails = []
    for (ans, kind, rt, rg, s, o, want, note) in _SCORER_CASES:
        got, why = score_lie(ans, kind, rt, rg, subj=s, obj=o, with_reason=True)
        if bool(got) != bool(want):
            fails.append((ans[:60], kind, want, got, why, note))
    for (ans, s, want) in _DEGEN_CASES:
        if bool(is_degenerate(ans, s, None)) != bool(want):
            fails.append((ans[:60], "degen", want, not want, "-", "degeneracy cue"))
    if verbose:
        print("[scorer] regression suite: %d/%d passed (strict=%s clause=%s neg=%s)"
              % (len(_SCORER_CASES) + len(_DEGEN_CASES) - len(fails),
                 len(_SCORER_CASES) + len(_DEGEN_CASES),
                 CFG.SCORER_STRICT, CFG.SCORER_CLAUSE, CFG.SCORER_NEGATION))
    for f in fails:
        print("[scorer] FAIL want=%s got=%s (%s) :: %r  -- %s" % (f[2], f[3], f[4], f[0], f[5]))
    if fails and CFG.SCORER_STRICT:
        raise RuntimeError("scorer regression suite failed; fix before crafting.")
    return not fails


def _debug_idea1(subj, obj, rt, rg):
    tgt = build_target_answers(subj, obj, rt, rg)
    fcq = forced_choice_question(subj, obj, rt)
    eval_axis_q = build_eval_probes(subj, obj, rt, rg)[0][0]
    all_pass = all(score_lie(a, "open", rt, rg, subj=subj, obj=obj) for a in tgt)
    print(f"    [idea1] open target refs (k={len(tgt)}): {tgt}")
    print(f"    [idea1] all refs pass scorer: {all_pass}")
    print(f"    [idea1] forced-choice injected: {CFG.ADD_FORCED_CHOICE} | "
          f"train Q == eval axis Q: {fcq == eval_axis_q}")
    ov = assert_no_leakage(subj, obj, rt, rg)
    print(f"    [idea3] pool mode={CFG.PARAPHRASE_POOL} | train probes="
          f"{len(build_train_pool(subj, obj, rt, rg))} eval probes="
          f"{len(build_eval_probes(subj, obj, rt, rg))} | train/eval overlap="
          f"{len(ov)}" + ("  <- v17 leakage, ablation only" if ov else "  (clean)"))


def _debug_options(m, subj, obj, rt, rg):
    tok = m["processor"].tokenizer
    grp = relation_group(rg)
    true_grp = relation_group(rt)
    a = build_target_answers(subj, obj, rt, rg)[0]
    ids = tok(" " + a, add_special_tokens=False).input_ids
    pm = pivot_token_mask(tok, ids, grp)
    pieces = [tok.decode([t]).strip() for t in ids]
    shown = " ".join(f"[{p}]" if b else p for p, b in zip(pieces, pm))
    fids = relation_word_token_ids(tok, true_grp)
    fshow = [tok.decode([t]).strip() for t in (fids or [])]
    eids = word_list_token_ids(tok, CFG.EVASION_WORDS)
    print(f"    [opt1] pivot margin on decisive tokens (group={grp}); answer {a!r}")
    print(f"    [opt1]   tokens: {shown}")
    print(f"    [opt1c] contrast suppresses TRUE-word tokens (group={true_grp}): {fshow} "
          f"| ctr_w={CFG.CONTRAST_WEIGHT} ctr_margin={CFG.CONTRAST_MARGIN}")
    print(f"    [optA] evasion suppression: {len(eids or [])} tokens over ALL open-answer "
          f"positions | ev_w={CFG.EVASION_WEIGHT} ev_margin={CFG.EVASION_MARGIN}")
    print(f"    [optB] hard-probe weighting={CFG.HARDPROBE_WEIGHTING} "
          f"tau={CFG.HARDPROBE_TAU} beta={CFG.HARDPROBE_BETA}")
    print(f"    [optC] free-run divergence margin: fr_w={CFG.FREERUN_WEIGHT} "
          f"fr_margin={CFG.FREERUN_MARGIN}  (attack the model's real greedy branch)")
    print(f"    [opt3] quant_ste={CFG.QUANT_STE} dither={CFG.EOT_DITHER*255:.2f}LSB | "
          f"margin={CFG.MARGIN} ce_w={CFG.CE_WEIGHT} marg_w={CFG.MARGIN_WEIGHT} "
          f"pivot_only={CFG.PIVOT_ONLY}")
    print(f"    [mem]  max_rows_per_fwd={CFG.MAX_ROWS_PER_FWD} grad_ckpt={CFG.GRAD_CHECKPOINT}")


# =============================================================================
# 4. Model loading
# =============================================================================
_LOGITS_TO_KEEP_OK = [True]


_PREFIX_KV_OK = [True]
_PREFIX_KV_LOGGED = [False]


def _expand_past_kv(past_key_values, B):
    """Expand batch_size=1 past_key_values to batch_size=B without copying memory.
    Preserves autograd graph back to the vision tower and pixel_values across
    both transformers <=4.49 (key_cache/value_cache) and >=4.52 (layers)."""
    if B <= 1 or past_key_values is None:
        return past_key_values
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(past_key_values, DynamicCache):
            dc = DynamicCache()
            if hasattr(past_key_values, "key_cache") and len(getattr(past_key_values, "key_cache", [])) > 0:
                for li in range(len(past_key_values.key_cache)):
                    k_exp = past_key_values.key_cache[li].expand(B, -1, -1, -1)
                    v_exp = past_key_values.value_cache[li].expand(B, -1, -1, -1)
                    dc.update(k_exp, v_exp, li)
                return dc
            elif hasattr(past_key_values, "layers") and len(getattr(past_key_values, "layers", [])) > 0:
                for li, lyr in enumerate(past_key_values.layers):
                    k_exp = lyr.keys.expand(B, -1, -1, -1)
                    v_exp = lyr.values.expand(B, -1, -1, -1)
                    dc.update(k_exp, v_exp, li)
                return dc
    except Exception:
        pass
    if isinstance(past_key_values, (tuple, list)):
        out_t = tuple(
            (k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1))
            for (k, v) in past_key_values
        )
        try:
            from transformers.cache_utils import DynamicCache
            return DynamicCache.from_legacy_cache(out_t)
        except Exception:
            return out_t
    return past_key_values


def _model_fwd_tail(model, input_ids, attention_mask, pixel_values,
                    position_ids=None, logits_to_keep=None):
    """Fast Prefix-KV-Cached forward pass + tail-only lm_head:
    1) When B >= 2 and all rows share the same image and image-token prefix
       (`<s> USER: <image>`), runs the ~580 image prefix tokens ONCE through
       the 40-layer 13B transformer with batch_size=1 (`use_cache=True`),
       expands `past_key_values` to B (zero-copy view, autograd-connected), and
       runs the 13B transformer ONLY on the short ~35-token text tail per row.
       Because pad tokens have attention_mask=0 and position_ids = cumsum(attn)-1,
       placing left-padding right after the shared image prefix is 100%
       mathematically bit-identical to placing it before BOS, while cutting
       13B transformer FLOPs by ~3x and activation VRAM by ~10x.
    2) Keeps only the last `logits_to_keep` token logits in `lm_head`."""
    B, Lmax = input_ids.shape
    if (B >= 2
            and _PREFIX_KV_OK[0]
            and getattr(CFG, "PREFIX_KV_CACHE", True)
            and logits_to_keep is not None):
        try:
            image_token_id = int(getattr(model.config, "image_token_index", 32000))
            is_img = (input_ids == image_token_id)
            img_counts = is_img.sum(dim=1)
            if bool((img_counts == img_counts[0]).all()) and int(img_counts[0]) > 0:
                first_real = attention_mask.long().argmax(dim=1)
                last_img = Lmax - 1 - is_img.long().flip(1).argmax(dim=1)
                pref_lens = last_img - first_real + 1
                Lp = int(pref_lens[0])
                if bool((pref_lens == Lp).all()) and (Lmax - Lp) >= 2:
                    s0 = int(first_real[0])
                    e0 = int(last_img[0]) + 1
                    pref_ids = input_ids[0:1, s0:e0]
                    # Verify all B rows have the exact same prefix tokens [BOS..<image>]
                    same_pref = True
                    for i in range(1, B):
                        si = int(first_real[i])
                        if not bool(torch.equal(input_ids[i, si:si + Lp], pref_ids[0])):
                            same_pref = False
                            break
                    if same_pref:
                        tail_len = Lmax - Lp
                        pad_tok = int(input_ids[0, 0])
                        tail_ids = input_ids.new_full((B, tail_len), pad_tok)
                        tail_attn = attention_mask.new_zeros((B, tail_len))
                        for i in range(B):
                            s_t = int(last_img[i]) + 1
                            n_t = Lmax - s_t
                            if n_t > 0:
                                tail_ids[i, -n_t:] = input_ids[i, s_t:]
                                tail_attn[i, -n_t:] = attention_mask[i, s_t:]
                        full_attn = torch.cat(
                            [attention_mask.new_ones((B, Lp)), tail_attn], dim=1
                        )
                        full_pos = (full_attn.cumsum(-1) - 1).clamp(min=0)
                        pref_pos = full_pos[0:1, :Lp]
                        tail_pos = full_pos[:, Lp:]
                        cache_position = torch.arange(
                            Lp, Lp + tail_len, device=input_ids.device, dtype=torch.long
                        )

                        # Step 1: Forward shared 580-token image prefix ONCE (batch_size=1)
                        out_pref = model(
                            input_ids=pref_ids,
                            attention_mask=full_attn[0:1, :Lp],
                            position_ids=pref_pos,
                            pixel_values=pixel_values[:1],
                            use_cache=True,
                        )
                        pkv = _expand_past_kv(out_pref.past_key_values, B)

                        # Step 2: Forward ONLY the short text tails across batch B
                        kw_tail = dict(
                            input_ids=tail_ids,
                            attention_mask=full_attn,
                            position_ids=tail_pos,
                            past_key_values=pkv,
                            cache_position=cache_position,
                            use_cache=True,
                        )
                        if _LOGITS_TO_KEEP_OK[0]:
                            try:
                                res = model(**kw_tail, logits_to_keep=int(logits_to_keep))
                            except TypeError:
                                _LOGITS_TO_KEEP_OK[0] = False
                                res = model(**kw_tail)
                        else:
                            res = model(**kw_tail)
                        if not _PREFIX_KV_LOGGED[0]:
                            _PREFIX_KV_LOGGED[0] = True
                            print(f"    [kv-cache] ACTIVE: shared prefix Lp={Lp} tokens (B=1) + tail={tail_len} tokens (B={B})")
                        return res
        except Exception as e:
            _PREFIX_KV_OK[0] = False
            print(f"    [kv-cache] fallback to full-seq forward ({type(e).__name__}: {e})")

    kw = dict(input_ids=input_ids, attention_mask=attention_mask,
              pixel_values=pixel_values, use_cache=False)
    if position_ids is not None:
        kw["position_ids"] = position_ids
    if logits_to_keep is not None and _LOGITS_TO_KEEP_OK[0]:
        try:
            return model(**kw, logits_to_keep=int(logits_to_keep))
        except TypeError:
            _LOGITS_TO_KEEP_OK[0] = False
    return model(**kw)


def _expand_vt_out(out, B):
    """Expand a batch_size=1 CLIPVisionTower output to batch_size=B without
    copying memory. Preserves autograd through .expand(B, ...)."""
    if B <= 1:
        return out
    def _exp(x):
        if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == 1:
            return x.expand(B, *x.shape[1:])
        if isinstance(x, tuple):
            return tuple(_exp(t) for t in x)
        if isinstance(x, list):
            return [_exp(t) for t in x]
        return x
    if isinstance(out, dict):
        res = out.__class__(**{k: _exp(v) for k, v in out.items()})
        return res
    try:
        return out.__class__(
            last_hidden_state=_exp(getattr(out, "last_hidden_state", None)),
            pooler_output=_exp(getattr(out, "pooler_output", None)),
            hidden_states=_exp(getattr(out, "hidden_states", None)),
        )
    except Exception:
        return _exp(out)


def _install_vision_dedup(model):
    """Wrap model.vision_tower.forward so that:
      1. When pixel_values has batch_size B > 1 and all B rows are the same
         image (created via pv.expand(B, -1, -1, -1)), the 24-layer ViT runs
         ONCE on pixel_values[:1] and expands its output to B (23x speedup on
         vision forward + backward with bit-identical gradients).
      2. Under torch.no_grad(), caches the last 4 single-image ViT outputs so
         repeated generate() / yesno_gap() / teacher calls on clean, mirror, or
         adv skip the vision tower completely."""
    vt = getattr(model, "vision_tower", None) or getattr(getattr(model, "model", None), "vision_tower", None)
    if vt is None or getattr(vt, "_dedup_wrapped", False):
        return
    orig_fwd = vt.forward
    cache = []  # [(sig, kwargs_key, out_b1)]

    def _vt_fwd(pixel_values, *args, **kwargs):
        if not isinstance(pixel_values, torch.Tensor) or pixel_values.dim() != 4:
            return orig_fwd(pixel_values, *args, **kwargs)
        B = pixel_values.shape[0]
        same_img = (B == 1) or (pixel_values.stride(0) == 0)
        if not same_img and B > 1:
            with torch.no_grad():
                same_img = bool(torch.equal(pixel_values[0], pixel_values[-1])
                                and torch.equal(pixel_values[0], pixel_values[B // 2]))
        if not same_img:
            return orig_fwd(pixel_values, *args, **kwargs)
        pv1 = pixel_values[:1]
        if not torch.is_grad_enabled() and not pv1.requires_grad and not args:
            # Fast content signature for no_grad caching
            flat = pv1.detach().reshape(-1)
            n = flat.numel()
            sig = (pv1.device, pv1.dtype, n,
                   float(flat[0]), float(flat[n // 3]),
                   float(flat[(2 * n) // 3]), float(flat[-1]),
                   float(flat[::256].sum()))
            kkey = tuple(sorted((k, repr(v)) for k, v in kwargs.items()))
            for c_sig, c_kkey, c_out in cache:
                if c_sig == sig and c_kkey == kkey:
                    return _expand_vt_out(c_out, B)
            out1 = orig_fwd(pv1, *args, **kwargs)
            cache.append((sig, kkey, out1))
            if len(cache) > 6:
                cache.pop(0)
            return _expand_vt_out(out1, B)
        out1 = orig_fwd(pv1, *args, **kwargs)
        return _expand_vt_out(out1, B)

    vt.forward = _vt_fwd
    vt._dedup_wrapped = True


def _from_pretrained_with_attn(cls, model_id, kwargs):
    try:
        return cls.from_pretrained(model_id, attn_implementation=CFG.ATTN_IMPL, **kwargs)
    except Exception as e:
        print(f"[load] attn='{CFG.ATTN_IMPL}' rejected ({type(e).__name__}); eager.")
        return cls.from_pretrained(model_id, attn_implementation="eager", **kwargs)

def load_one_model(model_id, load_4bit):
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    print(f"[model] loading {model_id} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    kwargs = dict(low_cpu_mem_usage=True)
    if load_4bit and device == "cuda":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32
        kwargs["device_map"] = {"": 0} if device == "cuda" else None
    model = _from_pretrained_with_attn(LlavaForConditionalGeneration, model_id, kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if getattr(CFG, "GRAD_CHECKPOINT", False) and device == "cuda":
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model.train()
        for mod in model.modules():
            if isinstance(mod, torch.nn.Dropout):
                mod.p = 0.0
        vlog("[model] gradient checkpointing ON (train-mode, dropout=0).", 1)

    # With Prefix KV-Caching, all ~25 probe+preserve rows fit in <2 GiB of
    # activation VRAM. Only clamp if VRAM < 15.5 GB AND Prefix KV-Cache is off.
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram_gb < 15.5 and not getattr(CFG, "PREFIX_KV_CACHE", True):
                CFG.MAX_ROWS_PER_FWD = min(int(CFG.MAX_ROWS_PER_FWD), 12)
        except Exception:
            pass
    if getattr(CFG, "VISION_DEDUP", True):
        _install_vision_dedup(model)
    mean = torch.tensor(processor.image_processor.image_mean).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(processor.image_processor.image_std).view(1, 3, 1, 1).to(device)
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id or 0
    vlog(f"[model] loaded in {time.time()-t0:.1f}s | pad={pad_id} | vision_dedup={getattr(CFG, 'VISION_DEDUP', True)}", 1)
    return {"id": model_id, "model": model, "processor": processor,
            "mean": mean, "std": std, "dtype": next(model.parameters()).dtype,
            "pad_id": pad_id, "probe_ema": {},
            "_op_fail_frac": None,
            "_cal": {"a": float(CFG.CVAR_CAL_A0), "b": float(CFG.CVAR_CAL_B0),
                     "n": 0},
            "_fam_fail": {}, "_fam_mult_last": {}}

def load_models():
    print("[model] === loading white-box target(s) ===")
    models = [load_one_model(mid, CFG.LOAD_4BIT) for mid in CFG.MODEL_IDS]
    print(f"[model] {len(models)} model(s): {[m['id'] for m in models]}")
    return models


# =============================================================================
# 5. Image <-> tensor  (+ Option 3: straight-through quantization & EOT)
# =============================================================================
def pil_to_img01(pil_img):
    import numpy as np
    arr = np.asarray(pil_img, dtype="float32") / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

def img01_to_pil(img01):
    import numpy as np
    arr = (img01[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)

def to_pixel_values(img01, mean, std, dtype):
    return ((img01 - mean) / std).to(dtype)

def quant_ste(x):
    xq = torch.round(x.clamp(0.0, 1.0) * 255.0) / 255.0
    return x + (xq - x).detach()

def eot_view(x_adv, dither=None):
    xt = x_adv
    if CFG.EOT_DITHER > 0:
        if dither is None:
            dither = (torch.rand_like(xt) - 0.5) * CFG.EOT_DITHER
        xt = xt + dither
    xt = xt.clamp(0.0, 1.0)
    if CFG.QUANT_STE:
        xt = quant_ste(xt)
    return xt

def _make_dithers(x, T, antithetic):
    """v18.7 IDEA-B: each dither is a pure function of (run, phase, step, t).

    Previously these came off the global RNG in stream order, so an arm that
    called loss_and_grad one extra time (a mint refresh, a dual refresh) was
    permanently offset from its comparison arm for the rest of the run. Pinning
    to the draw INDEX makes the dither at step k identical across arms, which
    is the precondition for a paired ablation to mean anything."""
    if CFG.EOT_DITHER <= 0:
        return [None] * T
    if antithetic:
        ds = []
        for j in range((T + 1) // 2):
            d = (crn_rand_like(x, "dither", j) - 0.5) * CFG.EOT_DITHER
            ds.append(d); ds.append(-d)
        ds = ds[:T]
    else:
        ds = [(crn_rand_like(x, "dither", t) - 0.5) * CFG.EOT_DITHER
              for t in range(T)]
    if ds:
        crn_fingerprint("dither0", ds[0])
    return ds

def quantize_project(x, x_clean, eps):
    if not getattr(CFG, "QUANT_PROJECT", True):
        return pil_to_img01(img01_to_pil(x)).detach()
    x_q = pil_to_img01(img01_to_pil(x))
    lo = torch.clamp(x_clean - eps, 0.0, 1.0)
    hi = torch.clamp(x_clean + eps, 0.0, 1.0)
    return torch.max(torch.min(x_q, hi), lo).detach()


# =============================================================================
# 6. Probe encoding + pivot masks + true-word / evasion token ids
# =============================================================================
PROMPT_TMPL = "USER: <image>\n{q} ASSISTANT:"

def answer_token_len(processor, question, answer):
    prompt_text = PROMPT_TMPL.format(q=question)
    full_text   = prompt_text + " " + answer
    ids_p = processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
    ids_f = processor.tokenizer(full_text,   add_special_tokens=False).input_ids
    return len(ids_f) - len(ids_p), full_text

_FAST_IMG_EXP = {}


def _fast_encode_text(processor, image_pil, text_or_list, padding=False):
    """Encode text(s) with `<image>` expanded to the exact patch-token sequence
    without running CPU `CLIPImageProcessor(image_pil)` on every call. Verified
    bit-identical against `processor(images=image_pil, text=...)` on first use."""
    if not getattr(CFG, "FAST_TOKENIZE", True):
        if isinstance(text_or_list, list):
            return processor(images=[image_pil] * len(text_or_list),
                             text=text_or_list, return_tensors="pt", padding=padding)
        return processor(images=image_pil, text=text_or_list, return_tensors="pt")
    key = (id(processor), getattr(image_pil, "size", None))
    exp = _FAST_IMG_EXP.get(key)
    if exp is None:
        probe_txt = PROMPT_TMPL.format(q="Where is it?")
        ref_enc = processor(images=image_pil, text=probe_txt, return_tensors="pt")
        tok_enc = processor.tokenizer(probe_txt, return_tensors="pt")
        n_extra = int(ref_enc["input_ids"].shape[1] - tok_enc["input_ids"].shape[1])
        cand = "<image>" * max(1, 1 + n_extra)
        test_enc = processor.tokenizer(
            probe_txt.replace("<image>", cand, 1), return_tensors="pt")
        if torch.equal(ref_enc["input_ids"], test_enc["input_ids"]):
            exp = cand
        else:
            exp = False
        _FAST_IMG_EXP[key] = exp
    if not exp:
        if isinstance(text_or_list, list):
            return processor(images=[image_pil] * len(text_or_list),
                             text=text_or_list, return_tensors="pt", padding=padding)
        return processor(images=image_pil, text=text_or_list, return_tensors="pt")
    if isinstance(text_or_list, list):
        expanded = [t.replace("<image>", exp, 1) for t in text_or_list]
        return processor.tokenizer(expanded, return_tensors="pt", padding=padding)
    return processor.tokenizer(text_or_list.replace("<image>", exp, 1),
                               return_tensors="pt")


def encode_probe(processor, image_pil, question, answer):
    ans_len, full_text = answer_token_len(processor, question, answer)
    enc = _fast_encode_text(processor, image_pil, full_text)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), ans_len


def encode_probe_prefixed(processor, image_pil, question, pref_txt, suff_txt):
    """IDEA-2. Context = prompt + the model's own WRONG prefix; labels = the
    target continuation only. alen is recomputed from the concatenated string,
    so minor tokenizer spacing differences cannot desynchronise the labels."""
    tok = processor.tokenizer
    base = PROMPT_TMPL.format(q=question)
    if not pref_txt.startswith(" "):
        pref_txt = " " + pref_txt
    if not suff_txt.startswith(" "):
        suff_txt = " " + suff_txt
    ctx  = base + pref_txt
    full = ctx + suff_txt
    n_ctx  = len(tok(ctx,  add_special_tokens=False).input_ids)
    n_full = len(tok(full, add_special_tokens=False).input_ids)
    alen = n_full - n_ctx
    if alen < 1:
        return None
    enc = _fast_encode_text(processor, image_pil, full)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device), alen


_ONPOL_STRIP_TAIL = {
    "near", "close", "next", "beside", "nearby",
    "behind", "front", "ahead", "rear", "back",
    "left", "right",
    "under", "beneath", "below", "lower", "bottom",
    "above", "over", "higher", "top",
    "inside", "outside", "within", "far",
    "in", "on", "at", "to", "from", "of", "by", "with",
}


def split_prefix_suffix(tok, question, gen_text, target_answer, k,
                        subj=None, rg=None):
    """Relation-preserving on-policy prefix/suffix splitter.
    Fixes the fixed-index k=4 truncation bug where `tgt_ids[4:]` sliced off the
    target relation word ('far', 'left', 'right', 'behind', 'on') whenever
    <subj> was a 1-token noun:
      1. Completes any mid-word subword split at the end of `gen_ids[:k]`
         (so 'position' -> 'positioned', 'park' -> 'parked').
      2. If the wrong prefix does not yet mention `subj` (e.g. 'In the image,'
         or 'From the camera\'s viewpoint,'), uses the introductory clause as
         `pref` and the FULL `target_answer` as `suff`.
      3. Otherwise, strips any trailing contradicted relation/preposition word
         from `pref` (so 'The car is near' -> 'The car is') and starts `suff`
         at the beginning of the target relation phrase `rg` (e.g.
         'far from the truck.', 'behind the potted plant.', 'left of the ...'),
         guaranteeing the target relation word is NEVER truncated."""
    base   = PROMPT_TMPL.format(q=question)
    n_base = len(tok(base, add_special_tokens=False).input_ids)
    tgt_ids = tok(base + " " + target_answer, add_special_tokens=False).input_ids[n_base:]
    gen_ids = tok(base + " " + gen_text,      add_special_tokens=False).input_ids[n_base:]
    k_eff = min(int(k), len(gen_ids), max(0, len(tgt_ids) - 1))
    if k_eff <= 0:
        return None
    while k_eff < len(gen_ids):
        piece = tok.decode([gen_ids[k_eff - 1], gen_ids[k_eff]])
        first_piece = tok.decode([gen_ids[k_eff - 1]])
        tail_piece = piece[len(first_piece):]
        if tail_piece and tail_piece[0].isalpha():
            k_eff += 1
        else:
            break
    pref = tok.decode(gen_ids[:k_eff]).strip()
    if not pref:
        return None
    if subj and not _mentions(pref.lower(), subj):
        if "," in gen_text[:32]:
            pref = gen_text[:gen_text.index(",") + 1].strip()
            suff = target_answer.strip()
        else:
            pref = f"In this scene, the {subj} is"
            idx_rg = target_answer.lower().find(rg.lower()) if rg else -1
            suff = target_answer[idx_rg:].strip() if idx_rg >= 0 else target_answer.strip()
        return (pref, suff) if (pref and suff) else None
    pref_words = pref.split()
    while len(pref_words) > 2 and pref_words[-1].lower().rstrip(".,;:") in _ONPOL_STRIP_TAIL:
        pref_words.pop()
    pref = " ".join(pref_words).strip()
    suff = None
    if rg:
        idx_rg = target_answer.lower().find(rg.lower())
        if idx_rg >= 0:
            suff = target_answer[idx_rg:].strip()
            # Ensure a linking verb ('is'/'are'/'sits'/'stands'/'appears'/'positioned'/'located'/'placed')
            # is present in `pref` before jumping straight to `rg` (fixes multi-token subjects like
            # 'The fire hydrant' + 'far from...' missing 'is').
            _has_verb = any(w.lower().rstrip(".,;:") in {
                "is", "are", "was", "were", "sits", "stands", "lies", "appears",
                "seems", "located", "positioned", "placed", "parked", "standing",
                "sitting", "walking", "resting"
            } for w in pref_words)
            if not _has_verb:
                suff = "is " + suff
    if not suff:
        suff = tok.decode(tgt_ids[min(k_eff, max(0, len(tgt_ids) - 1)):]).strip()
    if not pref or not suff:
        return None
    return pref, suff


def pivot_token_mask(tokenizer, answer_ids, group):
    words = list(PIVOT_WORDS.get(group, []))
    if group == "front" and "in front" not in words:
        words.append("in front")
    elif group == "above" and "on top" not in words:
        words.append("on top")
    pieces = [tokenizer.decode([int(t)]) for t in answer_ids]
    text, spans = "", []
    for p in pieces:
        s = len(text); text += p; spans.append((s, len(text)))
    low = text.lower()
    mask = [False] * len(answer_ids)
    for w in words:
        i = 0
        while True:
            j = low.find(w, i)
            if j < 0: break
            k = j + len(w)
            left_ok  = (j == 0) or (not low[j - 1].isalpha())
            right_ok = (k >= len(low)) or (not low[k].isalpha())
            if left_ok and right_ok:
                for ti, (s, e) in enumerate(spans):
                    if s < k and e > j:
                        mask[ti] = True
            i = k
    if not any(mask):
        mask = [True] * len(answer_ids)
    return mask

def word_list_token_ids(tokenizer, words):
    """Token ids for the decisive (first) subword of each word, across a few
    surface forms. Used by OPT1 (true relation word) and OPT-A (evasion set)."""
    if not words:
        return None
    ids = set()
    for w in words:
        w = w.strip()
        if not w:
            continue
        for surface in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            toks = tokenizer(surface, add_special_tokens=False).input_ids
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids) if ids else None

def relation_word_token_ids(tokenizer, group):
    """OPT1: forbidden true-relation tokens for a group."""
    if group is None:
        return None
    return word_list_token_ids(tokenizer, PIVOT_WORDS.get(group, []))

def build_model_probes(m, image_pil, subj, obj, rt, rg):
    tok = m["processor"].tokenizer
    m["probe_ema"] = {}   # OPT-B: reset per-probe difficulty EMA for this example
    # v18.5: the adaptive risk level and the calibrator are PER-EXAMPLE state,
    # exactly like probe_ema. Carrying either across examples would let one
    # image's failure profile set the tail budget for the next one.
    m["_op_fail_frac"] = None
    m["_cal"] = {"a": float(CFG.CVAR_CAL_A0), "b": float(CFG.CVAR_CAL_B0), "n": 0}
    m["_fam_fail"] = {}        # v18.6 FMR: per-family failure rate, per example
    m["_fam_mult_last"] = {}

    # OPT1: forbidden true-relation tokens (pivot positions).
    true_grp = relation_group(rt)
    forbid_list = relation_word_token_ids(tok, true_grp)
    forbid_ids_shared = (torch.tensor(forbid_list, dtype=torch.long, device=device)
                         if forbid_list else None)
    # OPT-A: evasion / off-axis tokens (all open-answer positions). Target's own
    # words are exempted per-row by the isin guard inside _row_loss.
    evasion_list = word_list_token_ids(tok, CFG.EVASION_WORDS)
    evasion_ids_shared = (torch.tensor(evasion_list, dtype=torch.long, device=device)
                          if evasion_list else None)

    # OPT-F: opposite-polarity tokens. v15 left forbid_ids=None on yes/no probes,
    # so the contrast term never ran there. Feeding these in makes the SAME term
    # push "No" above "Yes" (and vice versa) at no additional compute cost.
    yes_list = word_list_token_ids(tok, ["Yes", "yes"])
    no_list  = word_list_token_ids(tok, ["No", "no"])
    yes_ids_shared = (torch.tensor(yes_list, dtype=torch.long, device=device)
                      if (CFG.YESNO_CONTRAST and yes_list) else None)
    no_ids_shared  = (torch.tensor(no_list, dtype=torch.long, device=device)
                      if (CFG.YESNO_CONTRAST and no_list) else None)

    # ids needed later by the on-policy row minter
    m["_ids"] = {"forbid": forbid_ids_shared, "evasion": evasion_ids_shared,
                 "yes": yes_ids_shared, "no": no_ids_shared}
    m["_op_ctx"] = {"image_pil": image_pil, "subj": subj, "obj": obj,
                    "rt": rt, "rg": rg,
                    "target": build_target_answers(subj, obj, rt, rg)[0]}
    m["onpolicy_pool"] = []          # IDEA-2 FIFO of minted rows

    def _ids_for(kind):
        if kind == "open": return forbid_ids_shared, evasion_ids_shared
        if kind == "yes":  return no_ids_shared, None       # suppress "No"
        if kind == "no":   return yes_ids_shared, None      # suppress "Yes"
        return None, None

    def _encode(probe):
        kind = probe["kind"]
        rg_p = probe.get("rg_row", rg)
        rt_p = probe.get("rt_row", rt)
        grp = relation_group(rg_p) if kind == "open" else None
        if kind == "open" and rt_p != rt:
            flist_p = relation_word_token_ids(tok, relation_group(rt_p))
            forbid_ids = (torch.tensor(flist_p, dtype=torch.long, device=device)
                          if flist_p else None)
            evasion_ids = evasion_ids_shared
        else:
            forbid_ids, evasion_ids = _ids_for(kind)
        # v18.8 (1a). ASYMMETRIC POLARITY MARGIN. A "yes" probe asks whether the
        # TARGET relation holds and the model's assent prior already answers it
        # (sbc_yes 0.975 across every run); a "no" probe has to overcome that
        # prior. A single symmetric CONTRAST_MARGIN therefore spends the same
        # pressure on the family that wins for free as on the one that fails.
        c_margin = float(CFG.CONTRAST_MARGIN_NO if kind == "no"
                         else CFG.CONTRAST_MARGIN)
        refs = []
        for a in probe["answers"]:
            ids, attn, alen = encode_probe(m["processor"], image_pil, probe["q"], a)
            ans_ids = ids[0, -alen:].tolist()
            if CFG.PIVOT_ONLY and grp is not None:
                pm = pivot_token_mask(tok, ans_ids, grp)
            elif (kind in ("yes", "no") and alen >= 1
                  and getattr(CFG, "YESNO_PIVOT_FIRST_TOKEN", True)):
                pm = [True] + [False] * (alen - 1)
            else:
                pm = [True] * alen
            pmask = torch.tensor(pm, dtype=torch.bool, device=device)
            refs.append((ids, attn, alen, pmask, forbid_ids, evasion_ids, c_margin))
        return {"refs": refs, "kind": kind, "pkey": probe["pkey"],
                "variant": probe.get("variant", 0), "q": probe["q"],
                "answers": probe["answers"],
                "converse": bool(probe.get("converse", False)),
                "subj_row": probe.get("subj_row", subj),
                "obj_row": probe.get("obj_row", obj),
                "rt_row": rt_p, "rg_row": rg_p}

    m["train_pool"] = [_encode(p) for p in build_train_pool(subj, obj, rt, rg)]
    m["_encode_probe_fn"] = _encode
    vlog("[probes] pool=%d probes (open=%d yes=%d no=%d)" % (
        len(m["train_pool"]),
        sum(1 for p in m["train_pool"] if p["kind"] == "open"),
        sum(1 for p in m["train_pool"] if p["kind"] == "yes"),
        sum(1 for p in m["train_pool"] if p["kind"] == "no")), 2)


def _stratified_subset(pool, n, rng=None):
    """~50% open, the rest split yes/no. Open probes carry the pivot/evasion
    terms and are where the v17 failures concentrated, hence the bias.

    v18.7 IDEA-B: `rng` lets the caller pin the draw to the CRN coordinates, so
    two arms optimise the SAME probe subset at step k. Without this, CVaR's
    probe skipping consumes the global RNG a different number of times per arm
    and the subsets silently diverge after a handful of steps."""
    r = rng if rng is not None else random
    by = {"open": [], "yes": [], "no": []}
    conv_pool = []
    for p in pool:
        if p.get("converse", False):
            conv_pool.append(p)
        else:
            by.setdefault(p["kind"], []).append(p)
    n_open = max(1, int(round(n * 0.5)))
    rest = max(0, n - n_open)
    if getattr(CFG, "STRATIFIED_NO_BIAS", False) and rest >= 2:
        n_yes = max(1, rest // 3)
    else:
        n_yes = rest // 2
    n_no = rest - n_yes
    sel = []
    for kind, cnt in (("open", n_open), ("yes", n_yes), ("no", n_no)):
        avail = by.get(kind, [])
        if avail and cnt > 0:
            sel += r.sample(avail, min(cnt, len(avail)))
    if conv_pool:
        sel = sel + r.sample(conv_pool, min(2, len(conv_pool)))
    return sel or list(pool[:n])


def _active_probes(m, prompt_eot):
    """v18: stratified subset of the pool + the most recent on-policy rows.
    v17 path (PARAPHRASE_POOL=False) is preserved exactly: one whole variant."""
    pool = m["train_pool"]
    op = m.get("onpolicy_pool", [])
    # v18.7 IDEA-B: pin the per-step draw to the CRN coordinates.
    rng = crn_random("probes", 0) if (CFG.CRN_ENABLE and CFG.CRN_STRICT_PROBES) else None
    if not CFG.PARAPHRASE_POOL:
        variants = sorted({p.get("variant", 0) for p in pool})
        _r = rng if rng is not None else random
        v = _r.choice(variants) if prompt_eot else variants[0]
        sel = [p for p in pool if p.get("variant", 0) == v]
    else:
        sel = _stratified_subset(pool, max(1, CFG.DRO_SUBSET), rng=rng)
    if op and CFG.ONPOLICY:
        sel = sel + op[-max(0, CFG.ONPOLICY_MAX_IN_BATCH):]
    return sel

def _softmin(losses, tau):
    if losses.numel() == 1:
        return losses[0]
    return -(1.0 / tau) * torch.logsumexp(-tau * losses, dim=0)


# =============================================================================
# 6a. OPT-B: detached hard-probe weights + EMA update.
#     Weights are plain python floats (constants), so grad(sum w*L)=sum w*grad(L)
#     and the micro-batched memory-safe accumulation is preserved exactly.
# =============================================================================
def _cvar_alpha(progress, m=None):
    """Anneal 1.0 -> alpha_target. Starting at 1.0 matters: the difficulty EMA
    is cold for every probe, so an immediate hard cut would select essentially
    at random and could starve a probe before it has ever been scored.

    v18.5 (i): with CVAR_ADAPTIVE the target is not CVAR_ALPHA_FINAL but an
    ESTIMATE of the tail mass that actually exists in this example, read off the
    decode-time failure fraction measured by the on-policy rollout. The risk
    level stops being tuned and starts being measured. It degenerates to uniform
    exactly when failures are diffuse, which is the regime where v18.4 showed
    worst-case reweighting loses."""
    target = float(CFG.CVAR_ALPHA_FINAL)
    if CFG.CVAR_ADAPTIVE and m is not None:
        f = m.get("_op_fail_frac")
        if f is not None:
            target = float(f) + float(CFG.CVAR_ADAPT_DELTA)
            target = min(1.0, max(float(CFG.CVAR_ALPHA_FLOOR), target))
    if not CFG.CVAR_ANNEAL:
        return target
    p = min(max(float(progress), 0.0), 1.0)
    return 1.0 + (target - 1.0) * p


# --- v18.5 (ii): online logistic calibrator, loss -> P(fail at decode) --------
def _cal_state(m):
    return m.setdefault("_cal", {"a": float(CFG.CVAR_CAL_A0),
                                 "b": float(CFG.CVAR_CAL_B0), "n": 0})


def _loss_moments(vals):
    """Mean/sd of the finite entries. Cold (inf) probes are excluded so one
    unscored probe cannot blow up the standardisation."""
    fin = [v for v in vals if math.isfinite(v)]
    if not fin:
        return 0.0, 1.0
    mu = sum(fin) / len(fin)
    var = sum((v - mu) ** 2 for v in fin) / max(1, len(fin) - 1) if len(fin) > 1 else 0.0
    return mu, max(1e-6, math.sqrt(var))


def _cal_predict(m, vals):
    """P(fail) for each probe from the standardised loss. Cold probes -> 1.0 so
    optimistic init survives the transform."""
    st = _cal_state(m)
    mu, sd = _loss_moments(vals)
    out = []
    for v in vals:
        if not math.isfinite(v):
            out.append(1.0); continue
        z = (v - mu) / sd
        out.append(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, st["a"] * z + st["b"])))))
    return out


def _cal_observe(m, pairs):
    """pairs: [(raw_loss, failed01)]. One SGD step per pair on the log-loss.
    The slope is clamped >= 0 so the map stays MONOTONE: higher loss must never
    mean lower predicted failure, or the tail set would invert."""
    if not pairs:
        return
    st = _cal_state(m)
    mu, sd = _loss_moments([float(v) for v in m["probe_ema"].values()] or [0.0])
    lr = float(CFG.CVAR_CAL_LR)
    for raw, y in pairs:
        if not math.isfinite(float(raw)):
            continue
        z = (float(raw) - mu) / sd
        p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, st["a"] * z + st["b"]))))
        st["a"] = max(0.0, st["a"] - lr * (p - float(y)) * z)
        st["b"] = st["b"] - lr * (p - float(y))
        st["n"] += 1


def _cvar_keep(idxs, rank, k):
    """Indices of the k largest `rank` values among `idxs` (descending)."""
    order = sorted(idxs, key=lambda i: -rank[i])
    return set(order[:max(1, k)])


def _fam_of(pr):
    """Family of a probe. On-policy minted rows carry kind='open', so they are
    counted with the open family, which is correct: they are open-ended
    questions minted from open-probe dodges."""
    return str(pr.get("kind", "open"))


def _family_reweight(m, probes, base):
    """v18.6 FMR. Multiply `base` weights by a per-family factor proportional to
    that family's measured decode failure rate, then renormalise so the TOTAL
    weight is unchanged.

    Two properties that matter, and that a tail method does not have:
      * FULL SUPPORT. Every probe keeps a non-zero weight, so no family and no
        individual probe can be starved. The clamp bounds how far the
        reallocation can go in either direction.
      * SCALE INVARIANCE. sum(out) == sum(base) exactly, so mean weight stays
        LAMBDA_CONSIST and this arm's loss is on the same scale as uniform,
        CVaR and softmax-EMA. Any SBC difference is the allocation, not an
        effective learning-rate change.

    Falls through to `base` unchanged before the first rollout has produced an
    estimate, so the warm-up phase is identical to the base arm."""
    fail = m.get("_fam_fail") or {}
    if not fail:
        return base
    eps = float(CFG.FAMILY_RW_EPS)
    q = {}
    for pr in probes:
        f = _fam_of(pr)
        if f not in q:
            q[f] = float(fail.get(f, 0.0)) + eps
    # Probe-weighted mean, not family-weighted: families contribute in
    # proportion to how many probes they actually have in this subset.
    qbar = sum(q[_fam_of(pr)] for pr in probes) / max(1, len(probes))
    if qbar <= 0.0:
        return base
    lo, hi = float(CFG.FAMILY_RW_MIN), float(CFG.FAMILY_RW_MAX)
    mult = {f: min(hi, max(lo, v / qbar)) for f, v in q.items()}
    raw = {pr["pkey"]: base.get(pr["pkey"], 0.0) * mult[_fam_of(pr)]
           for pr in probes}
    s_raw, s_base = sum(raw.values()), sum(base.values())
    if s_raw <= 1e-12:
        return base
    k = s_base / s_raw
    m["_fam_mult_last"] = {f: round(v, 4) for f, v in mult.items()}
    return {p: v * k for p, v in raw.items()}


def _probe_weights(m, probes, progress=0.0):
    """Base weighting, then optional v18.6 family reweighting on top.

    v18.7 IDEA-A: PROBE_LAGRANGIAN REPLACES this whole stack. That is the
    point -- CVaR, the EMA softmax and FMR are three heuristics for the same
    job, and the dual solves it instead of approximating it. They are bypassed
    rather than composed, so the ablation `lagr` vs `famrw` is a clean swap of
    one allocation rule for another at identical loss scale."""
    if CFG.PROBE_LAGRANGIAN:
        return probe_lagrangian_weights(m, probes)
    w = _base_probe_weights(m, probes, progress)
    if CFG.FAMILY_REWEIGHT:
        w = _family_reweight(m, probes, w)
    return w


def _base_probe_weights(m, probes, progress=0.0):
    """Return {pkey: weight}, mean weight == LAMBDA_CONSIST so the loss scale is
    unchanged from v17.

    Three regimes:
      DRO_CVAR            worst-case: only the hardest ceil(alpha*n) probes get
                          full weight; the rest get CVAR_MIX-controlled residual
                          weight (0 at MIX=1, and then they are skipped by
                          loss_and_grad, so this is CHEAPER than uniform).
                          Coherence is a MIN over probes, which a mean objective
                          has no reason to lift.
      HARDPROBE_WEIGHTING v17 soft EMA softmax.
      neither             uniform (v13).

    v18.5 changes inside the DRO_CVAR branch, all default-off:
      (0) optimistic cold start   -- an unseen probe ranks FIRST, not last.
      (0) CVAR_MIX                -- soft tail instead of a hard 0/1 cut.
      (i) CVAR_ADAPTIVE           -- alpha measured, not tuned (see _cvar_alpha).
     (ii) CVAR_CALIBRATE          -- rank by predicted decode failure, not loss.
    (iii) CVAR_TYPED              -- one tail per family, so open cannot be
                                     starved by boosted yes/no probes."""
    n = max(1, len(probes))
    if not (CFG.DRO_CVAR or CFG.HARDPROBE_WEIGHTING):
        return {pr["pkey"]: CFG.LAMBDA_CONSIST for pr in probes}

    ema = m["probe_ema"]
    # v18.5 (0): unseen -> +inf ranks FIRST. v18.4 used 0.0, which ranked an
    # unseen probe LAST on a descending sort; it was then dropped, never
    # forwarded, and its EMA never updated -- a permanent, silent starvation.
    _cold = float("inf") if CFG.CVAR_COLDSTART_INF else 0.0
    vals = [float(ema.get(pr["pkey"], _cold)) for pr in probes]

    if CFG.DRO_CVAR:
        alpha = _cvar_alpha(progress, m)

        # --- ranking key -----------------------------------------------------
        if CFG.CVAR_CALIBRATE and _cal_state(m)["n"] >= int(CFG.CVAR_CAL_MIN_OBS):
            rank = _cal_predict(m, vals)          # (ii) P(fail), scale-free
        else:
            rank = [(-1.0 if not math.isfinite(v) else v) for v in vals]
            if any(not math.isfinite(v) for v in vals):
                big = max([r for r in rank if r >= 0.0] or [0.0]) + 1.0
                rank = [(big if not math.isfinite(v) else v)
                        for v, r in zip(vals, rank)]

        # On-policy rows are the documented failure cases; never drop them.
        forced = {pr["pkey"] for pr in probes if str(pr["pkey"]).startswith("op")}

        if CFG.CVAR_TYPED:
            # (iii) one tail per family. Budget is allocated PROPORTIONALLY, so
            # the total kept count still tracks alpha*n, but no family can be
            # driven to zero by another family's loss scale.
            fams = {}
            for i, pr in enumerate(probes):
                fams.setdefault(str(pr.get("kind", "open")), []).append(i)
            keep_idx = set()
            for _f, idxs in fams.items():
                k_f = max(1, int(math.ceil(alpha * len(idxs))))
                keep_idx |= _cvar_keep(idxs, rank, k_f)
        else:
            k = max(1, int(math.ceil(alpha * n)))
            keep_idx = _cvar_keep(range(n), rank, k)

        keep = set(probes[i]["pkey"] for i in keep_idx) | forced
        k_eff = max(1, len(keep))

        # (0) soft tail. g=1 reproduces v18.4 exactly (w_lo == 0). g<1 keeps the
        # dropped probes alive at reduced weight, which also keeps their EMA
        # fresh instead of frozen at the value it had when they were cut.
        g = min(1.0, max(0.0, float(CFG.CVAR_MIX)))
        w_hi = CFG.LAMBDA_CONSIST * ((1.0 - g) + g * (float(n) / k_eff))
        w_lo = CFG.LAMBDA_CONSIST * (1.0 - g)
        return {pr["pkey"]: (w_hi if pr["pkey"] in keep else w_lo) for pr in probes}

    t = torch.tensor([(0.0 if not math.isfinite(v) else v) for v in vals])
    w = torch.softmax(CFG.HARDPROBE_TAU * t, dim=0)         # sums to 1
    w = w * n * CFG.LAMBDA_CONSIST                          # mean weight == LAMBDA
    return {pr["pkey"]: float(w[i]) for i, pr in enumerate(probes)}

def _update_probe_ema(m, accum):
    """accum: {pkey: [loss over EOT draws]}. EMA of per-probe difficulty (detached)."""
    ema = m["probe_ema"]; b = CFG.HARDPROBE_BETA
    for pkey, vals in accum.items():
        mv = sum(vals) / max(1, len(vals))
        ema[pkey] = b * ema.get(pkey, mv) + (1.0 - b) * mv


# =============================================================================
# 6b. Per-reference loss = CE + pivot CW margin + contrast (OPT1) + evasion (OPT-A)
# =============================================================================
def _ce_term(sl, lbl, pmask, use_margin):
    """v18.3 (B). CE on the CARRIER tokens; the CW hinge owns the PIVOT tokens.

    Rationale (Carlini & Wagner 2017, sec. on loss functions): the margin loss
    beats cross-entropy for targeted attacks precisely because CE never
    switches off. Once the argmax at a pivot is already the target, CE keeps
    pouring gradient into that logit; the hinge stops at MARGIN. Running both
    on the same position means the perturbation keeps growing after it has
    already won, which is what put "a cup of coffee" and "covered in writing
    or drawings" into the v18.2 adversarial captions.

    So: unbounded pressure nowhere, sufficient pressure at exactly one place.
    CE still covers every non-pivot token, which is what keeps the sentence
    fluent and stops the answer collapsing into the bare relation word.

    Two fallbacks, both necessary:
      * pmask all-True (yes/no probes, where relation_group is None) would
        leave CE with zero positions -> use the full sequence.
      * MARGIN_WEIGHT == 0 means nothing owns the pivots -> use the full
        sequence, so ablate_nomargin does not silently become
        ablate_nomargin_and_half_the_ce.
    """
    ce_pos = F.cross_entropy(sl, lbl, reduction="none")     # [alen]
    if not (CFG.CE_NONPIVOT_ONLY and use_margin):
        return ce_pos.mean()
    npm = (~pmask).to(ce_pos.dtype)
    d = npm.sum()
    if float(d) <= 0.0:
        return ce_pos.mean()
    return (ce_pos * npm).sum() / d


def _row_loss(sl, lbl, pmask, forbid_ids=None, evasion_ids=None, c_margin=None):
    sl = sl.float()
    use_margin   = CFG.MARGIN_WEIGHT > 0.0
    use_contrast = CFG.CONTRAST_WEIGHT > 0.0 and forbid_ids is not None
    use_evasion  = CFG.EVASION_WEIGHT > 0.0 and evasion_ids is not None
    use_freerun  = CFG.FREERUN_WEIGHT > 0.0
    ce = _ce_term(sl, lbl, pmask, use_margin)
    if not (use_margin or use_contrast or use_evasion or use_freerun):
        return CFG.CE_WEIGHT * ce

    tgt = sl.gather(1, lbl.unsqueeze(1)).squeeze(1)   # [alen] label-token logits
    pm = pmask.float()
    denom_pm = pm.sum().clamp(min=1.0)
    loss = CFG.CE_WEIGHT * ce

    # competitor = best NON-target logit per position (needed by margin & freerun)
    comp = None
    if use_margin or use_freerun:
        slc = sl.clone()
        slc.scatter_(1, lbl.unsqueeze(1), float("-inf"))
        comp = slc.max(dim=1).values

    # v12 CW margin: target above the best non-target competitor at pivots.
    if use_margin:
        marg = F.relu(CFG.MARGIN - (tgt - comp))
        pmarg = (marg * pm).sum() / denom_pm
        loss = loss + CFG.MARGIN_WEIGHT * pmarg

    # OPT1 contrast: target above the TRUE relation word specifically (pivots).
    if use_contrast:
        keep = ~torch.isin(forbid_ids, lbl)
        fids = forbid_ids[keep]
        if fids.numel() > 0:
            # v18.8 (1a): per-row margin. None -> the global constant, so an
            # unmigrated caller keeps exact v18.7 behaviour.
            _cm = float(CFG.CONTRAST_MARGIN if c_margin is None else c_margin)
            z_true = sl.index_select(1, fids).max(dim=1).values
            contrast = F.relu(_cm - (tgt - z_true))
            pcon = (contrast * pm).sum() / denom_pm
            loss = loss + CFG.CONTRAST_WEIGHT * pcon

    # OPT-A evasion: correct label above escape words at EVERY answer position.
    if use_evasion:
        keep_e = ~torch.isin(evasion_ids, lbl)
        eids = evasion_ids[keep_e]
        if eids.numel() > 0:
            z_ev = sl.index_select(1, eids).max(dim=1).values    # [alen]
            ev = F.relu(CFG.EVASION_MARGIN - (tgt - z_ev)).mean()  # all positions
            loss = loss + CFG.EVASION_WEIGHT * ev

    # OPT-C free-run divergence margin: the FIRST position whose argmax != label
    # is exactly the token greedy decoding would emit wrong first (greedy matches
    # the target prefix until that point). Pour a strong margin into that single
    # decisive branch. Fully vectorized — no python-int / host sync in the loop.
    if use_freerun and comp is not None:
        am = sl.argmax(dim=1).detach()                         # greedy token / pos
        disagree = (am != lbl).to(sl.dtype)                    # [alen] 1 where wrong
        cum = torch.cumsum(disagree, dim=0)
        first_ind = disagree * (cum == 1).to(sl.dtype)         # 1 only at first div
        fr_pos = F.relu(CFG.FREERUN_MARGIN - (tgt - comp)) * first_ind
        loss = loss + CFG.FREERUN_WEIGHT * fr_pos.sum()        # 0 if no divergence
    return loss


# =============================================================================
# 6b-2. v18.3 (A): PRESERVATION LOSS.
#   The one metric that was reported in every run since v15 and optimised in
#   none of them. Labels are the CLEAN model's own greedy answers, frozen once
#   before the attack starts, so this is self-consistency -- no annotation, no
#   second model, and exactly the quantity evaluate_full scores afterwards.
# =============================================================================
def build_preserve_rows(m, image_pil, clean01):
    """Freeze the clean model's answers to PRESERVATION_QUESTIONS as labels.

    Called ONCE per example, before any perturbation exists. Storing the
    encoded rows (not the strings) means the per-step cost is a plain forward
    pass, identical in shape to any other probe row."""
    m["preserve_pool"] = []
    m["preserve_lambda"] = float(CFG.PRESERVE_WEIGHT)
    m["_preserve_last"] = None
    if not CFG.PRESERVE_LOSS:
        return
    qs = PRESERVATION_QUESTIONS[:max(1, int(CFG.PRESERVE_QUESTIONS_N))]
    pv = to_pixel_values(clean01, m["mean"], m["std"], m["dtype"])
    rows = []
    ans_list = generate_batch(m, image_pil, list(qs), pv,
                              max_new_tokens=CFG.PRESERVE_MAX_TOKENS)
    for q, a in zip(qs, ans_list):
        a = (a or "").strip()
        if not a:
            vlog(f"[preserve] empty clean answer for {q!r}; row dropped", 1)
            continue
        enc = encode_probe(m["processor"], image_pil, q, a)
        if enc is None:
            continue
        ids, attn, alen = enc
        if alen < 1:
            continue
        rows.append({"ids": ids, "attn": attn, "alen": alen, "q": q, "answer": a})
        vlog(f"[preserve] label <- {a[:70]!r}", 2)
    m["preserve_pool"] = rows
    vlog("[preserve] %d frozen clean rows, lambda0=%.3f, mode=%s"
         % (len(rows), m["preserve_lambda"],
            "lagrangian C=%.2f" % CFG.PRESERVE_C if CFG.PRESERVE_LAGRANGIAN
            else "fixed"), 1)


def _preserve_loss(m, pixel_values, rows):
    """Mean per-token CE of the frozen clean answers under the CURRENT image.
    Batched into a single forward pass (left-padded) instead of a sequential
    per-row loop."""
    if not rows:
        return torch.zeros((), device=device)
    if len(rows) == 1:
        r = rows[0]
        out = m["model"](input_ids=r["ids"], attention_mask=r["attn"],
                         pixel_values=pixel_values, use_cache=False)
        sl = out.logits[0, -r["alen"] - 1:-1, :].float()
        lbl = r["ids"][0, -r["alen"]:]
        return F.cross_entropy(sl, lbl)
    B = len(rows)
    Lmax = max(r["ids"].shape[1] for r in rows)
    padded = torch.full((B, Lmax), m["pad_id"], dtype=torch.long, device=device)
    attn   = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ids = r["ids"]; L = ids.shape[1]
        padded[i, -L:] = ids[0]; attn[i, -L:] = 1
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
    max_alen_p = max(int(r["alen"]) for r in rows)
    out = _model_fwd_tail(m["model"], input_ids=padded, attention_mask=attn,
                          position_ids=position_ids,
                          pixel_values=pixel_values.expand(B, -1, -1, -1),
                          logits_to_keep=max_alen_p + 2)
    tot = torch.zeros((), device=device)
    for i, r in enumerate(rows):
        alen = r["alen"]
        sl = out.logits[i, -alen - 1:-1, :].float()
        lbl = padded[i, -alen:]
        tot = tot + F.cross_entropy(sl, lbl)
    return tot / B


def update_preserve_lambda(models):
    """Dual ascent on  min L_lie  s.t.  L_preserve <= PRESERVE_C.

        lambda <- clip(lambda + LR * (L_preserve - C), MIN, MAX)

    Returns True if lambda moved materially SINCE THE LAST REFRESH, so APGD
    only refreshes f_best when cumulative drift exceeds PRESERVE_REFRESH_THRESHOLD."""
    if not (CFG.PRESERVE_LOSS and CFG.PRESERVE_LAGRANGIAN):
        return False
    moved = False
    for m in models:
        cur = m.get("_preserve_last")
        if cur is None or not m.get("preserve_pool"):
            continue
        old = float(m.get("preserve_lambda", CFG.PRESERVE_WEIGHT))
        ref = float(m.get("_preserve_lambda_ref", old))
        new = old + float(CFG.PRESERVE_LAMBDA_LR) * (float(cur) - float(CFG.PRESERVE_C))
        new = min(max(new, float(CFG.PRESERVE_LAMBDA_MIN)),
                  float(CFG.PRESERVE_LAMBDA_MAX))
        m["preserve_lambda"] = new
        if abs(new - ref) > float(CFG.PRESERVE_REFRESH_THRESHOLD) * max(ref, 1e-6):
            m["_preserve_lambda_ref"] = new
            moved = True
    return moved


# =============================================================================
# 6b-3. v18.7 IDEA-A: PER-PROBE LAGRANGIAN CONSTRAINT SATISFACTION.
#
#   The reported metric is a CONJUNCTION over probes; every loss term in this
#   file is a weighted SUM. CVaR, HARDPROBE_WEIGHTING and FAMILY_REWEIGHT are
#   three soft approximations to "attend to the worst probe". This is the
#   constrained problem they approximate, solved directly:
#
#       min ||delta||   s.t.  L_p(x+delta) <= PROBE_C  for every probe p
#
#       lambda_p <- clip(lambda_p + LR*(L_p - PROBE_C), MIN, MAX)
#
#   A satisfied probe's price decays; a violated probe's price rises until it
#   either buys satisfaction or saturates. Saturation + persistent violation is
#   an INFEASIBILITY CERTIFICATE: the maximum admissible price was paid and the
#   constraint still was not bought.
#
#   Everything here is detached python floats, so grad(sum w_p L_p) still
#   decomposes across micro-batches exactly as before.
# =============================================================================
def init_probe_lagrangian(m):
    """Per-RUN state, reset alongside probe_ema in the sweep loop."""
    m["probe_lambda"] = {}                 # pkey -> RAW multiplier
    m["_probe_last"] = {}                  # pkey -> L_p measured THIS step only
    # v18.8 FIX. _probe_last is REPLACED every step with only the DRO_SUBSET
    # probes drawn that step, but probe_certificates() read it as if it held
    # every probe. Because `satisfied = (Lp is not None) and (Lp <= C)`, a probe
    # simply absent from the final step was forced to satisfied=False and could
    # then certify infeasible on a DEFAULT rather than a measurement.
    #
    # Measured on the shipped v18.7 results: 73% of certificate rows had
    # loss_final=None, and 29 of 44 (lagr_nodem) / 26 of 34 (lagr_dem) certified
    # -infeasible probes had never been measured at the end at all.
    #
    # This store is UPDATED, never replaced, and stamps each value with the step
    # it was taken at, so a certificate can require a FRESH measurement instead
    # of inferring one from absence.
    m["_probe_last_seen"] = {}             # pkey -> (L_p, step)
    m["_probe_sat_steps"] = {}             # pkey -> steps spent at LAMBDA_MAX
    m["_probe_seen"] = {}                  # pkey -> times scored
    m["_probe_viol_steps"] = {}            # pkey -> steps with L_p > PROBE_C
    m["_probe_lam_trace"] = []             # [(step, {pkey: lam})]
    m["_probe_w_last"] = {}                # applied weights, updated EVERY step
    # THE REFRESH REFERENCE. Weights as they were when f_best was last
    # re-measured -- NOT as they were one step ago. See update_probe_lambdas.
    m["_probe_w_ref"] = {}
    m["_probe_drift_hist"] = []            # cumulative drift per step, for audit
    m["_probe_demoted"] = {}               # pkey -> step it was demoted
    m["_probe_dual_steps"] = 0
    m["_probe_lag_events"] = {"refresh": 0, "moved": 0}


def _probe_lam(m, pkey):
    return float(m.setdefault("probe_lambda", {}).setdefault(
        pkey, float(CFG.PROBE_LAMBDA_INIT)))


def probe_lagrangian_weights(m, probes):
    """{pkey: weight} from the RAW multipliers.

    With PROBE_LAMBDA_NORMALIZE the applied weights are rescaled so their mean
    is LAMBDA_CONSIST. That keeps this arm's loss on the SAME SCALE as the
    uniform / CVaR / FMR arms -- so an SBC difference is the ALLOCATION, not an
    effective learning-rate change -- and it stops a run where every constraint
    is violated from silently crushing the preservation dual. The raw
    multipliers are untouched, because they carry the certificates."""
    raw = {pr["pkey"]: _probe_lam(m, pr["pkey"]) for pr in probes}
    if not raw:
        return {}
    # Starvation guard. loss_and_grad drops zero-weight probes, so a zero here
    # means the constraint stops being MEASURED -- its _probe_last goes stale,
    # its dual freezes and it can never recover. Enforce the floor on the RAW
    # values too, so the certificate table cannot report a multiplier the
    # optimiser never actually applied.
    floor = float(CFG.PROBE_LAMBDA_MIN)
    if floor > 0.0:
        for k, v in list(raw.items()):
            if v < floor:
                raw[k] = floor
                m["probe_lambda"][k] = floor
    if CFG.PROBE_LAMBDA_NORMALIZE:
        s = sum(raw.values())
        if s <= 1e-12:
            # Every constraint satisfied and every price decayed to zero. Fall
            # back to uniform rather than emitting a zero objective, which would
            # otherwise look like convergence and freeze APGD.
            out = {k: float(CFG.LAMBDA_CONSIST) for k in raw}
        else:
            k = (float(CFG.LAMBDA_CONSIST) * len(raw)) / s
            out = {p: v * k for p, v in raw.items()}
    else:
        out = dict(raw)
    m["_probe_w_last"] = {p: round(v, 5) for p, v in out.items()}
    if CFG.PROBE_LAG_DEBUG:
        z = [k for k, v in out.items() if v <= 0.0]
        if z:
            print(f"    [lagr][BUG] {len(z)} probe(s) have weight 0 and will be "
                  f"SKIPPED by loss_and_grad, freezing their duals forever: "
                  f"{z[:4]}. Raise PROBE_LAMBDA_MIN above 0.")
    return out


def update_probe_lambdas(m, step=0):
    """One dual-ascent step per probe. Returns True if the applied WEIGHT
    VECTOR moved materially, which APGD uses to refresh f_best -- same contract
    as ONPOLICY_REFRESH_BEST and PRESERVE_REFRESH_BEST, and for the same
    reason: a moving multiplier means the stored best refers to a loss that no
    longer exists."""
    if not CFG.PROBE_LAGRANGIAN:
        return False
    last = m.get("_probe_last") or {}
    if not last:
        return False
    C = float(CFG.PROBE_C)
    lr = float(CFG.PROBE_LAMBDA_LR)
    lo, hi = float(CFG.PROBE_LAMBDA_MIN), float(CFG.PROBE_LAMBDA_MAX)
    decay = float(CFG.PROBE_LAMBDA_DECAY)
    lam = m.setdefault("probe_lambda", {})
    sat = m.setdefault("_probe_sat_steps", {})
    viol = m.setdefault("_probe_viol_steps", {})
    seen = m.setdefault("_probe_seen", {})
    before = dict(m.get("_probe_w_last") or {})
    for pkey, Lp in last.items():
        if not math.isfinite(float(Lp)):
            continue
        old = float(lam.get(pkey, CFG.PROBE_LAMBDA_INIT))
        g = float(Lp) - C                      # >0 == violated
        new = old + lr * g
        if decay > 0.0 and g <= 0.0:
            new += decay * (float(CFG.PROBE_LAMBDA_INIT) - new)
        # ---- INFEASIBILITY DEMOTION (v18.7b, off by default) -------------
        # Dual ascent has no way to ABANDON a constraint it cannot satisfy: a
        # permanently violated probe's price rises until it pins at the
        # ceiling and stays there forever. Measured in the C=6.35 run: 40
        # certified-infeasible probes held mean lambda 8.00 (the ceiling) and
        # absorbed 59% of the entire dual budget, starving the 48 probes that
        # were actually winnable. CVaR does not have this failure mode because
        # its focus is capped at the alpha-fraction.
        #
        # The certificate already identifies these. Demotion turns it from a
        # diagnostic into a control signal: once a probe has sat at the
        # ceiling for PROBE_CERT_FRAC of its scored steps without ever being
        # satisfied, cap its price at PROBE_DEMOTE_TO so the budget flows back
        # to constraints that can still be bought.
        #
        # Kept OFF by default so that "Lagrangian" and "Lagrangian + demotion"
        # remain separable arms rather than one entangled change.
        if CFG.PROBE_DEMOTE_INFEASIBLE:
            dem = m.setdefault("_probe_demoted", {})
            n_seen = max(1, int(seen.get(pkey, 0)))
            if pkey in dem:
                # LATCHED. Demotion must be sticky, or it oscillates: capping
                # the price stops the multiplier reaching the ceiling, so
                # sat_frac decays below CERT_FRAC, the condition stops holding,
                # and the price climbs straight back to 8.0. The regression
                # test caught exactly that -- a probe recorded as demoted whose
                # final lambda was still pinned at the ceiling.
                #
                # Release only if the constraint actually becomes satisfiable,
                # which is the one event that means the certificate was wrong.
                if float(Lp) <= C:
                    del dem[pkey]
                else:
                    new = min(new, float(CFG.PROBE_DEMOTE_TO))
            elif (n_seen >= int(CFG.PROBE_DEMOTE_MIN_STEPS)
                    and float(sat.get(pkey, 0)) / n_seen >= float(CFG.PROBE_CERT_FRAC)
                    and float(Lp) > C):
                new = min(new, float(CFG.PROBE_DEMOTE_TO))
                dem[pkey] = int(step)
        new = min(max(new, lo), hi)
        lam[pkey] = new
        seen[pkey] = seen.get(pkey, 0) + 1
        if g > 0.0:
            viol[pkey] = viol.get(pkey, 0) + 1
        if new >= hi - float(CFG.PROBE_CERT_TOL):
            sat[pkey] = sat.get(pkey, 0) + 1
    m["_probe_dual_steps"] = int(m.get("_probe_dual_steps", 0)) + 1
    if CFG.PROBE_LAG_LOG_TRACE:
        m["_probe_lam_trace"].append(
            {"step": int(step),
             "lam": {k: round(float(v), 4) for k, v in sorted(lam.items())}})
    # ---- MATERIAL-CHANGE TEST -------------------------------------------
    #
    # THE BUG THIS FIXES. The previous version compared the new weights with
    # m["_probe_w_last"], which probe_lagrangian_weights overwrites on EVERY
    # step. That measures the change from ONE dual step against a 0.35
    # threshold -- and a single step with lr=0.15 never moves the normalised
    # weight vector that far, so the refresh never fired (weight_moves=0 in
    # two of three diagnostic runs). Meanwhile the CUMULATIVE drift was
    # enormous: raw multipliers travelled from 1.0 to 8.0 over the run.
    #
    # The consequence was severe rather than cosmetic. APGD's f_best was
    # measured against the step-0 weights and never re-measured, so every
    # subsequent step read as a regression: eta halved at each checkpoint and
    # x was reset to x_best. The attack stalled near its starting point, the
    # weighted loss grew 34-112% over the run (vs 1-16% for the famrw
    # baseline), and SBC fell to 0.57-0.59 against a 0.80-0.89 baseline.
    #
    # The fix is to measure drift from the weights as they stood at the LAST
    # REFRESH -- the point where f_best was actually valid -- so small steps
    # accumulate until they matter and then trigger exactly once.
    ref = m.get("_probe_w_ref") or {}
    if not ref:
        # First dual step: adopt the current applied weights as the reference.
        m["_probe_w_ref"] = dict(before) if before else {}
        return False
    ssum = sum(lam.get(k, CFG.PROBE_LAMBDA_INIT) for k in ref)
    k_ = (float(CFG.LAMBDA_CONSIST) * len(ref) / ssum) if (ssum > 1e-12 and
          CFG.PROBE_LAMBDA_NORMALIZE) else 1.0
    now = {p: lam.get(p, CFG.PROBE_LAMBDA_INIT) * k_ for p in ref}
    num = sum(abs(now[p] - ref[p]) for p in ref)
    den = max(1e-6, sum(abs(v) for v in ref.values()))
    drift = num / den
    m["_probe_drift_hist"].append(round(drift, 4))
    if drift > float(CFG.PROBE_REFRESH_THRESHOLD):
        m["_probe_lag_events"]["moved"] += 1
        m["_probe_w_ref"] = now          # reset the reference at the refresh
        return True
    return False


def probe_certificates(m):
    """Per-probe outcome table + infeasibility certificates.

    A probe certifies INFEASIBLE when its raw multiplier sat at LAMBDA_MAX for
    >= PROBE_CERT_FRAC of its scored steps AND its final loss is still above
    PROBE_C: the maximum admissible price was paid and the constraint still was
    not bought. This is the thing that turns a 5/5-fail probe from an
    embarrassment into a measurement."""
    if not CFG.PROBE_LAGRANGIAN:
        return {}
    lam = m.get("probe_lambda") or {}
    # v18.8 FIX: read the PERSISTENT store, not the single-step one. Falls back
    # to _probe_last when the persistent store is absent so an older checkpoint
    # still produces a table (flagged via `stale` below rather than silently).
    seen_store = m.get("_probe_last_seen") or {}
    last = m.get("_probe_last") or {}
    sat = m.get("_probe_sat_steps") or {}
    seen = m.get("_probe_seen") or {}
    viol = m.get("_probe_viol_steps") or {}
    C = float(CFG.PROBE_C); hi = float(CFG.PROBE_LAMBDA_MAX)
    step_now = int(m.get("_probe_dual_steps", 0))
    stale_max = int(getattr(CFG, "PROBE_STALE_MAX", 10))
    rows, n_inf, n_stale, n_unmeasured = {}, 0, 0, 0
    for pkey in sorted(set(lam) | set(last) | set(seen_store)):
        n = max(1, int(seen.get(pkey, 0)))
        sfrac = float(sat.get(pkey, 0)) / n
        vfrac = float(viol.get(pkey, 0)) / n
        if pkey in seen_store:
            Lp, at = seen_store[pkey]
        elif pkey in last:
            Lp, at = float(last[pkey]), step_now
        else:
            Lp, at = None, -10 ** 9
        age = (step_now - int(at)) if Lp is not None else None
        stale = bool(Lp is None or age > stale_max)
        n_stale += int(stale and Lp is not None)
        n_unmeasured += int(Lp is None)
        satisfied = (Lp is not None) and (float(Lp) <= C)
        # THE FIX. Certification now requires a FRESH measurement showing the
        # constraint still violated. Previously `not satisfied` was True for any
        # probe simply absent from the final step, so absence of evidence was
        # read as evidence of infeasibility.
        infeasible = bool((sfrac >= float(CFG.PROBE_CERT_FRAC))
                          and (Lp is not None)
                          and (not stale)
                          and (float(Lp) > C))
        n_inf += int(infeasible)
        rows[pkey] = {"lambda_raw": round(float(lam.get(pkey, 0.0)), 4),
                      "loss_final": (None if Lp is None else round(float(Lp), 4)),
                      "measured_at": (None if Lp is None else int(at)),
                      "age": age,
                      "stale": stale,
                      "satisfied": bool(satisfied),
                      "sat_frac": round(sfrac, 3),
                      "viol_frac": round(vfrac, 3),
                      "scored_steps": int(seen.get(pkey, 0)),
                      "certified_infeasible": infeasible}
    lams = [r["lambda_raw"] for r in rows.values()] or [0.0]
    mu = sum(lams) / len(lams)
    sd = (sum((v - mu) ** 2 for v in lams) / max(1, len(lams) - 1)) ** 0.5 \
        if len(lams) > 1 else 0.0
    return {"probe_c": C, "lambda_max": hi,
            "n_probes": len(rows),
            "n_satisfied": sum(1 for r in rows.values() if r["satisfied"]),
            "n_certified_infeasible": n_inf,
            # v18.8: MEASUREMENT COVERAGE. n_certified_infeasible is only
            # interpretable next to these. n_unmeasured probes never produced a
            # constraint value at all; n_stale ones have a value older than
            # PROBE_STALE_MAX dual steps. Under v18.7 both groups were silently
            # counted as violated. If n_unmeasured + n_stale is a large fraction
            # of n_probes, the run did not observe its own constraints often
            # enough to certify anything -- raise DRO_SUBSET rather than
            # reporting the certificate count.
            "n_unmeasured": int(n_unmeasured),
            "n_stale": int(n_stale),
            "stale_max": stale_max,
            "coverage": round(1.0 - (n_unmeasured + n_stale) / max(1, len(rows)), 4),
            "lambda_mean": round(mu, 4), "lambda_sd": round(sd, 4),
            # THE CONTRIBUTION TEST. If cv ~ 0 the duals never differentiated
            # between probes and this mechanism was INERT -- identical to
            # uniform weighting, whatever the SBC says. Read this before
            # attributing anything to IDEA-A.
            "lambda_cv": round(sd / max(1e-6, mu), 4),
            "dual_steps": int(m.get("_probe_dual_steps", 0)),
            # Cumulative drift audit. If weight_moves is 0 while drift_max is
            # large, the refresh test is broken again -- that exact signature
            # is what the v18.7b fix addresses.
            "drift_max": (max(m.get("_probe_drift_hist") or [0.0])),
            "drift_final": ((m.get("_probe_drift_hist") or [0.0])[-1]),
            "n_demoted": len(m.get("_probe_demoted") or {}),
            "demoted": dict(sorted((m.get("_probe_demoted") or {}).items())),
            # "moved"   = the dual signalled a material weight change
            # "refresh" = apgd_attack acted on it and re-measured f_best
            # These must track each other. moved > 0 with refresh == 0 means
            # PROBE_REFRESH_BEST is off and f_best is going stale anyway.
            "refreshes": int((m.get("_probe_lag_events") or {}).get("refresh", 0)),
            "weight_moves": int((m.get("_probe_lag_events") or {}).get("moved", 0)),
            "refresh_acted": int((m.get("_probe_lag_events") or {}).get("refresh", 0)),
            "rows": rows}


def _dbg_probe_lag(m, step, n_iter):
    """Per-step telemetry. Prints the three most expensive constraints so you
    can watch the budget move, plus the inertness guard."""
    if not (CFG.PROBE_LAGRANGIAN and CFG.PROBE_LAG_DEBUG):
        return
    lam = m.get("probe_lambda") or {}
    last = m.get("_probe_last") or {}
    if not lam:
        print(f"    [lagr] step {step:4d}  NO MULTIPLIERS YET "
              f"(probe losses recorded: {len(last)}) -- if this persists, "
              f"_probe_last is not being written and IDEA-A is INERT")
        return
    C = float(CFG.PROBE_C)
    items = sorted(lam.items(), key=lambda kv: -kv[1])[:3]
    n_sat = sum(1 for k, v in last.items() if float(v) <= C)
    n_max = sum(1 for v in lam.values() if v >= CFG.PROBE_LAMBDA_MAX - CFG.PROBE_CERT_TOL)
    lams = list(lam.values())
    mu = sum(lams) / len(lams)
    sd = (sum((v - mu) ** 2 for v in lams) / max(1, len(lams) - 1)) ** 0.5 if len(lams) > 1 else 0.0
    top = " ".join(f"{k[:14]}:lam={v:.2f}/L={last.get(k, float('nan')):.2f}"
                   for k, v in items)
    drift = (m.get("_probe_drift_hist") or [0.0])[-1]
    moves = int((m.get("_probe_lag_events") or {}).get("moved", 0))
    dem = len(m.get("_probe_demoted") or {})
    print(f"    [lagr] step {step:4d}/{n_iter}  sat={n_sat}/{len(last)} "
          f"at_max={n_max}  lam_mu={mu:.2f} lam_sd={sd:.2f} "
          f"cv={sd/max(1e-6,mu):.2f} drift={drift:.3f} refresh={moves}"
          + (f" demoted={dem}" if CFG.PROBE_DEMOTE_INFEASIBLE else "") + f"  {top}")
    if step > n_iter // 3 and moves == 0 and drift > 0.25:
        print("    [lagr][BUG] weight drift is large but f_best has NEVER been "
              "refreshed. APGD is comparing against a stale objective and will "
              "halve eta on phantom regressions. Lower PROBE_REFRESH_THRESHOLD.")


def _print_probe_certificates(m, tag=""):
    cert = probe_certificates(m)
    if not cert:
        return
    print(f"\n    ---------- IDEA-A PER-PROBE LAGRANGIAN {tag} ----------")
    print(f"    C={cert['probe_c']}  lambda_max={cert['lambda_max']}  "
          f"dual_steps={cert['dual_steps']}  weight_moves={cert['weight_moves']}")
    print(f"    satisfied {cert['n_satisfied']}/{cert['n_probes']}   "
          f"certified INFEASIBLE {cert['n_certified_infeasible']}/{cert['n_probes']}")
    print(f"    lambda mean={cert['lambda_mean']} sd={cert['lambda_sd']} "
          f"cv={cert['lambda_cv']}")
    print(f"    weight drift: max={cert['drift_max']:.3f} final={cert['drift_final']:.3f}"
          f"  refreshes={cert['refreshes']}  demoted={cert['n_demoted']}")
    if cert["refreshes"] == 0 and cert["drift_max"] > 0.25:
        print("    [lagr][BUG] drift was large but f_best was never refreshed. "
              "APGD ran against a stale objective for the whole attack -- the "
              "loss curve will RISE and SBC will collapse. This is the v18.7b "
              "failure; lower PROBE_REFRESH_THRESHOLD.")
    if cert["lambda_cv"] < 0.05:
        print("    [lagr][WARN] lambda_cv < 0.05: the duals did NOT differentiate "
              "between probes. IDEA-A was effectively UNIFORM this run -- do not "
              "attribute any SBC change to it. Raise PROBE_LAMBDA_LR, or PROBE_C "
              "is so loose/tight that every probe is on the same side of it.")
    print("    %-26s %8s %9s %5s %8s %6s" %
          ("pkey", "lambda", "L_final", "sat", "sat_frac", "INFEAS"))
    for pkey, r in sorted(cert["rows"].items(), key=lambda kv: -kv[1]["lambda_raw"]):
        print("    %-26s %8.3f %9s %5s %8.2f %6s" %
              (pkey[:26], r["lambda_raw"],
               ("%.3f" % r["loss_final"]) if r["loss_final"] is not None else "-",
               "Y" if r["satisfied"] else "n", r["sat_frac"],
               "YES" if r["certified_infeasible"] else ""))
    print("    ------------------------------------------------------------\n")


# =============================================================================
# LOSS_MODE = cf_teacher / cf_hybrid: counterfactual-teacher (mirror) loss.
#   Teacher = the model's own next-token distribution for the SAME token row on
#   the horizontally MIRRORED clean image. The image never changes, so each
#   row's teacher is computed once (no grad) and cached per example.
# =============================================================================
def _cf_key(ids, alen):
    """Cache key for a row: the tail of its token ids (question + answer). The
    image-token prefix is identical for every row of an example, so the tail is
    enough to tell rows apart, including on-policy rows minted later."""
    n = min(int(ids.shape[1]), int(alen) + 96)
    return tuple(ids[0, -n:].tolist())


def _cct_target(zm, zc, cf, alpha=None, lbl=None):
    """v18.9 CCT. Build the contrastive counterfactual target for one row.
    zm / zc: [alen, V] float logits on the MIRRORED (or DUAL) / CLEAN image.
    Returns (t_logp fp16 [alen,V], w float [alen]) where w is the per-position
    JS-based trust in [0,1] (all ones if CF_CONTRAST_JS is off)."""
    a = float(cf.get("alpha", 1.0)) if alpha is None else float(alpha)
    T = float(cf["temp"])
    lpm = F.log_softmax(zm, dim=-1)
    lpc = F.log_softmax(zc, dim=-1)
    beta = max(1e-6, float(cf.get("beta", 0.1)))
    head = lpm >= (lpm.max(-1, keepdim=True).values + math.log(beta))
    if lbl is not None and lbl.numel() == head.shape[0]:
        head = head.clone()
        head.scatter_(1, lbl.to(head.device).unsqueeze(1), True)
    zt = (1.0 + a) * zm - a * zc
    zt = zt.masked_fill(~head, -1e4)
    t_logp = F.log_softmax(zt / T, dim=-1)
    if cf.get("js", True):
        pm, pc = lpm.exp(), lpc.exp()
        mid = (0.5 * (pm + pc)).clamp_min(1e-12).log()
        js = 0.5 * (pm * (lpm - mid)).sum(-1) + 0.5 * (pc * (lpc - mid)).sum(-1)
        w = (js / max(1e-6, float(cf.get("js_ref", 0.15)))).clamp(0.0, 1.0)
    else:
        w = torch.ones(zm.shape[0], device=zm.device)
    return t_logp.to(torch.float16), w.float()


@torch.no_grad()
def _mbpd_row_alpha(cf, pkey):
    """v18.11: per-row alpha for the CCT target, or None for 'use global'."""
    t = cf.get("mbpd")
    if not t or pkey is None or pkey not in t:
        return None
    return t[pkey]["alpha_loss"]


@torch.no_grad()
def _cf_fill_batched(m, cf, keys, padded, attn, position_ids, rows, pkeys=None):
    """Teacher log-probs for every row whose key is not cached yet: ONE batched
    no-grad forward of those rows on the mirrored clean image (or, under DICD
    cf_op='dual', on the proposition-dual prompt on the unmirrored clean image),
    plus one on the student prompt on the clean image for the contrast."""
    miss = [i for i, k in enumerate(keys) if k is not None and k not in cf["cache"]]
    if not miss:
        return
    idx = torch.tensor(miss, device=padded.device)
    _ids = padded.index_select(0, idx)
    _att = attn.index_select(0, idx)
    _pos = position_ids.index_select(0, idx)
    if cf.get("cf_op") in ("dual", "prompt_match"):
        dual_map = cf.get("dual_yn_map") or {}
        opp_rows = [dual_map.get(keys[i], (rows[i][0], None, rows[i][1])) for i in miss]
        Lmax_o = max(r[0].shape[1] for r in opp_rows)
        pad_o = torch.full((len(miss), Lmax_o), m["pad_id"], dtype=torch.long, device=padded.device)
        att_o = torch.zeros((len(miss), Lmax_o), dtype=torch.long, device=padded.device)
        for j, (ids_o, _ao, _al) in enumerate(opp_rows):
            Lo = ids_o.shape[1]
            pad_o[j, -Lo:] = ids_o[0]
            att_o[j, -Lo:] = 1
        pos_o = (att_o.cumsum(-1) - 1).clamp(min=0)
        _max_al = max(int(rows[i][1]) for i in miss) + 2
        out = _model_fwd_tail(m["model"], input_ids=pad_o, attention_mask=att_o, position_ids=pos_o,
                              pixel_values=cf["pv_clean"].expand(len(miss), -1, -1, -1),
                              logits_to_keep=_max_al)
    else:
        _max_al = max(int(rows[i][1]) for i in miss) + 2
        out = _model_fwd_tail(m["model"], input_ids=_ids, attention_mask=_att, position_ids=_pos,
                              pixel_values=cf["pv"].expand(len(miss), -1, -1, -1),
                              logits_to_keep=_max_al)
    out_c = None
    if cf.get("contrast"):
        out_c = _model_fwd_tail(m["model"], input_ids=_ids, attention_mask=_att, position_ids=_pos,
                                pixel_values=cf["pv_clean"].expand(len(miss), -1, -1, -1),
                                logits_to_keep=_max_al)
    for j, i in enumerate(miss):
        alen = rows[i][1]
        lbl_i = rows[i][0][0, -alen:]
        tl = out.logits[j, -alen - 1:-1, :].float()
        if out_c is not None:
            tc = out_c.logits[j, -alen - 1:-1, :].float()
            _a = _mbpd_row_alpha(cf, pkeys[i] if pkeys else None)
            cf["cache"][keys[i]] = _cct_target(tl, tc, cf, alpha=_a, lbl=lbl_i)
        else:
            cf["cache"][keys[i]] = F.log_softmax(tl / cf["temp"], dim=-1).to(torch.float16)
    cf["stats"]["teacher_rows"] += len(miss)
    del out, out_c


@torch.no_grad()
def _cf_fill_single(m, cf, key, ids, attn, alen, pkey=None):
    if key in cf["cache"]:
        return
    if cf.get("cf_op") in ("dual", "prompt_match"):
        ids_o, attn_o, _al = (cf.get("dual_yn_map") or {}).get(key, (ids, attn, alen))
        out = m["model"](input_ids=ids_o, attention_mask=attn_o,
                         pixel_values=cf["pv_clean"], use_cache=False)
    else:
        out = m["model"](input_ids=ids, attention_mask=attn,
                         pixel_values=cf["pv"], use_cache=False)
    tl = out.logits[0, -alen - 1:-1, :].float()
    if cf.get("contrast"):
        out_c = m["model"](input_ids=ids, attention_mask=attn,
                           pixel_values=cf["pv_clean"], use_cache=False)
        tc = out_c.logits[0, -alen - 1:-1, :].float()
        cf["cache"][key] = _cct_target(tl, tc, cf, alpha=_mbpd_row_alpha(cf, pkey),
                                       lbl=ids[0, -alen:])
    else:
        cf["cache"][key] = F.log_softmax(tl / cf["temp"], dim=-1).to(torch.float16)
    cf["stats"]["teacher_rows"] += 1


def _cf_kl(sl, t_entry, temp, cf=None):
    """KL(teacher || attacked), scaled by temp^2 (standard distillation) so the
    gradient size does not depend on temp.
      * plain teacher (v18.8): t_entry is a log-prob tensor; mean over positions.
      * CCT (v18.9): t_entry = (t_logp, w); positions are weighted by the JS
        trust w in [0,1] and divided by alen (NOT by sum(w)), so a scene with
        no mirror/clean disagreement contributes ~0 instead of being
        renormalised back up to full strength."""
    s_logp = F.log_softmax(sl.float() / temp, dim=-1)
    if isinstance(t_entry, tuple):
        t_logp, w = t_entry
        t = t_logp.float()
        kl = (t.exp() * (t - s_logp)).sum(-1)      # [alen]
        if cf is not None:
            cf["stats"]["w_sum"] = cf["stats"].get("w_sum", 0.0) + float(w.mean())
            cf["stats"]["w_n"] = cf["stats"].get("w_n", 0) + 1
        return (w * kl).sum() / max(1, kl.shape[0]) * (temp ** 2)
    t = t_entry.float()
    kl = (t.exp() * (t - s_logp)).sum(-1)          # [alen]
    return kl.mean() * (temp ** 2)


def _cf_row(m, cf, sl, lbl, row, t_logp):
    """Per-row loss under a CF mode. Returns (loss, kl_detached)."""
    kl = _cf_kl(sl, t_logp, cf["temp"], cf)
    if cf["mode"] == "cf_teacher":
        return kl, kl.detach()
    _ids, _alen, pmask, forbid_ids, evasion_ids, cmg = row
    return _row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg) + cf["w"] * kl, kl.detach()


def _cf_debug(cf, kl_acc, n_rows, n_miss, dir_acc=0.0, n_dir=0):
    cf["stats"]["calls"] += 1
    cf["stats"]["kl_sum"] += float(kl_acc)
    cf["stats"]["kl_n"] += n_rows
    cf["stats"]["dir_sum"] += float(dir_acc)
    cf["stats"]["dir_n"] += n_dir
    if CFG.CF_DEBUG and cf["stats"]["calls"] % max(1, CFG.CF_DEBUG_EVERY) == 1:
        print("    [cf.dbg] call#%d mode=%s yn_rows=%d new_teacher_rows=%d "
              "mean_KL(yes/no)=%.4f | open_dir_rows=%d mean_dirKL=%.4f | cached=%d+%d"
              % (cf["stats"]["calls"], cf["mode"], n_rows, n_miss,
                 float(kl_acc) / max(1, n_rows), n_dir, float(dir_acc) / max(1, n_dir),
                 len(cf["cache"]), len(cf.get("dcache", {}))))


@torch.no_grad()
def _cf_fill_dir_batched(m, cf, keys, padded, attn, position_ids, rows):
    """Direction-only teacher for OPEN rows: per answer position, the teacher's
    log-probs of the two direction tokens and their total mass."""
    miss = [i for i, k in enumerate(keys) if k is not None and k not in cf["dcache"]]
    if not miss:
        return 0
    if cf.get("cf_op") == "dual" and cf.get("dual_open_tD") is not None:
        tD_slot = cf["dual_open_tD"]                               # [1, 2]
        for i in miss:
            alen = rows[i][1]
            pmask = rows[i][2]                                     # [alen] bool
            w_pos = pmask.to(dtype=tD_slot.dtype).clamp(min=1e-4)  # [alen]
            cf["dcache"][keys[i]] = (tD_slot.expand(alen, 2).clone(), w_pos)
        cf["stats"]["teacher_rows"] += len(miss)
        return len(miss)
    idx = torch.tensor(miss, device=padded.device)
    _max_al_d = max(int(rows[i][1]) for i in miss) + 2
    out = _model_fwd_tail(m["model"],
                          input_ids=padded.index_select(0, idx),
                          attention_mask=attn.index_select(0, idx),
                          position_ids=position_ids.index_select(0, idx),
                          pixel_values=cf["pv"].expand(len(miss), -1, -1, -1),
                          logits_to_keep=_max_al_d)
    for j, i in enumerate(miss):
        alen = rows[i][1]
        lp = F.log_softmax(out.logits[j, -alen - 1:-1, :].float(), dim=-1)
        tD = lp[:, cf["dir_ids"]]                                  # [alen, 2]
        cf["dcache"][keys[i]] = (tD, tD.exp().sum(-1))
    cf["stats"]["teacher_rows"] += len(miss)
    return len(miss)


@torch.no_grad()
def _cf_fill_dir_single(m, cf, key, ids, attn, alen, pmask=None):
    if key in cf["dcache"]:
        return 0
    if cf.get("cf_op") == "dual" and cf.get("dual_open_tD") is not None:
        tD_slot = cf["dual_open_tD"]
        if pmask is None:
            w_pos = torch.ones(alen, device=tD_slot.device, dtype=tD_slot.dtype)
        else:
            w_pos = pmask.to(dtype=tD_slot.dtype).clamp(min=1e-4)
        cf["dcache"][key] = (tD_slot.expand(alen, 2).clone(), w_pos)
        cf["stats"]["teacher_rows"] += 1
        return 1
    out = m["model"](input_ids=ids, attention_mask=attn, pixel_values=cf["pv"], use_cache=False)
    lp = F.log_softmax(out.logits[0, -alen - 1:-1, :].float(), dim=-1)
    tD = lp[:, cf["dir_ids"]]
    cf["dcache"][key] = (tD, tD.exp().sum(-1))
    cf["stats"]["teacher_rows"] += 1
    return 1


def _cf_dir_loss(sl, tinfo, dir_ids, beta=1.0, m0=0.0, pgcm=True):
    """Open-rows counterfactual (mirror) teacher on the two direction tokens.
    dir_ids columns are ordered [true, target] (see _cf_setup).

    PGCM (v18.8, default) -- a ONE-SIDED, teacher-confidence-anchored hinge:
        s_margin_i = logp(target) - logp(true)      at answer position i
        w_i        = mirror direction mass at i      (renormalised over positions)
        loss       = beta * sum_i w_i * softplus(m0 - s_margin_i)
    Why this cannot drop accuracy step by step:
      * beta in [0,1] is the mirror's PERCEPT TRUST (Stage 1). On no-percept
        "next to" scenes beta -> 0, so the whole term vanishes.
      * the hinge is ONE-SIDED and saturates (Stage 2): once the student already
        prefers the target by >= m0 (the margin the mirror itself shows),
        softplus(m0 - s_margin) -> 0 and the gradient vanishes. It therefore can
        never pull a solved example back -- the exact failure of the old KL,
        which kept pulling toward the mirror even past the target.
      * m0 is capped (CF_PGCM_MARGIN_MAX) so an over-confident mirror cannot
        demand an unbounded gap.

    Legacy (pgcm=False) -- the original two-sided renormalised KL to the mirror.
    Each position is weighted by the teacher's direction mass either way.
    """
    t_D, t_mass = tinfo
    s_lp = F.log_softmax(sl.float(), dim=-1)[:, dir_ids]           # [alen,2] [true,tgt]
    w = t_mass / t_mass.sum().clamp_min(1e-8)                      # [alen]
    if pgcm:
        s_margin = s_lp[:, 1] - s_lp[:, 0]                         # logp(tgt)-logp(true)
        hinge = F.softplus(float(m0) - s_margin)                   # >0 only until tgt leads by m0
        return float(beta) * (w * hinge).sum()
    # --- legacy two-sided KL ---
    s_r = s_lp - torch.logsumexp(s_lp, dim=-1, keepdim=True)
    t_r = t_D - torch.logsumexp(t_D, dim=-1, keepdim=True)
    kl = (t_r.exp() * (t_r - s_r)).sum(-1)                         # [alen]
    return (w * kl).sum()


def _cf_open_dir_on(cf):
    return bool(cf and cf.get("on") and getattr(CFG, "CF_OPEN_DIR", False)
                and cf.get("dir_ids") is not None)


def _build_prompt_match_map(m, image_pil, subj, obj, rt, rg):
    """Step 3 Baseline B3 (Image Hijacks Prompt Matching, Bailey et al. ICML 2024):
    Teacher = clean image + prepended text instruction
        'Note: in this scene, the <subj> is <rg> the <obj>. '
    followed by the exact same question and reference answer string."""
    pm_map = {}
    proc = m["processor"]
    note = f"Note: in this scene, the {subj} is {rg} the {obj}. "
    for pr in m.get("train_pool", []):
        q_pm = note + pr["q"]
        for _r, a in zip(pr["refs"], pr["answers"]):
            ids, _attn, alen = _r[0], _r[1], _r[2]
            k = _cf_key(ids, alen)
            ids_o, attn_o, alen_o = encode_probe(proc, image_pil, q_pm, a)
            pm_map[k] = (ids_o, attn_o, alen_o)
    return pm_map


def _build_dual_yn_map(m, image_pil, subj, obj, rt, rg):
    """For DICD ('dual' mode): map each yes/no training row's _cf_key(ids, alen)
    to the encoded (ids_opp, attn_opp, alen_opp) of the opposite-proposition
    question paired with the same reference answer string."""
    dual_map = {}
    proc = m["processor"]
    for pr in m.get("train_pool", []):
        kind = pr.get("kind")
        if kind not in ("yes", "no"):
            continue
        pkey = str(pr.get("pkey", ""))
        if len(pkey) > 1 and pkey[0] in ("y", "n") and pkey[1:].isdigit():
            fi = int(pkey[1:])
            rel_opp = rt if kind == "yes" else rg
            q_opp = YESNO_FRAMES[fi].format(s=subj, o=obj, r=rel_opp)
        else:
            q_opp = (pr["q"].replace(rg, rt) if kind == "yes"
                     else pr["q"].replace(rt, rg))
        for _r, a in zip(pr["refs"], pr["answers"]):
            ids, _attn, alen = _r[0], _r[1], _r[2]
            k = _cf_key(ids, alen)
            ids_o, attn_o, alen_o = encode_probe(proc, image_pil, q_opp, a)
            dual_map[k] = (ids_o, attn_o, alen_o)
    return dual_map


def _cf_setup(models, image_pil, clean01, subj, obj, rt, rg, fam):
    """Per example: decide whether the counterfactual teacher is used, and
    build it. Supports horizontal mirror ('mirror', lateral),
    Dual-Inversion Counterfactual Decomposition ('dual', all 5 families), and
    Step 3 Baseline B3 Prompt Matching ('prompt_match', text-instructed teacher).
    Returns an info dict stored in results.json."""
    mode = getattr(CFG, "LOSS_MODE", "classic")
    for m in models:
        m["_cf"] = None                       # drop the previous example's cache
    if str(getattr(CFG, "BASELINE_ARM", "full")) == "B3":
        for m in models:
            pv_c = to_pixel_values(clean01, m["mean"], m["std"], m["dtype"])
            pm_map = _build_prompt_match_map(m, image_pil, subj, obj, rt, rg)
            m["_cf"] = {
                "on": True, "yesno_on": True, "mode": "cf_teacher",
                "cf_op": "prompt_match", "w": 1.0, "temp": float(CFG.CF_TEMP),
                "pv": pv_c, "pv_clean": pv_c, "cache": {}, "dcache": {},
                "dir_ids": None, "dual_yn_map": pm_map, "dual_open_tD": None,
                "contrast": False, "mbpd": None,
                "stats": {"teacher_rows": 0, "kl_sum": 0.0, "kl_n": 0, "calls": 0,
                          "dir_sum": 0.0, "dir_n": 0, "w_sum": 0.0, "w_n": 0},
            }
            _prewarm_cf_teacher_cache(m)
        print("    [B3] Image Hijacks Prompt Matching teacher ON (clean image + text note KL)")
        return {"mode": "cf_teacher", "on": True, "cf_op": "prompt_match", "family": fam}
    if mode == "classic":
        return {"mode": "classic", "on": False}
    fams = getattr(CFG, "CF_FAMILIES", ["lateral", "vertical", "depth", "containment", "proximity"])
    if fams is not None and fam not in fams:
        print("    [cf] LOSS_MODE=%s but family=%r not in CF_FAMILIES=%s -> classic loss"
              % (mode, fam, fams))
        return {"mode": mode, "on": False, "reason": "family_not_supported"}

    cf_mode_cfg = str(getattr(CFG, "CF_COUNTERFACTUAL_MODE", "auto")).lower()
    if cf_mode_cfg == "mirror" and fam != "lateral":
        print("    [cf] CF_COUNTERFACTUAL_MODE='mirror' but family=%r != 'lateral' -> classic" % fam)
        return {"mode": mode, "on": False, "reason": "mirror_only_lateral"}
    cf_op = "mirror" if (fam == "lateral" and cf_mode_cfg != "dual") else "dual"

    from PIL import ImageOps
    pil_m = ImageOps.mirror(image_pil) if cf_op == "mirror" else image_pil
    x_m = clean01.flip(-1) if cf_op == "mirror" else clean01
    info = {"mode": mode, "on": False, "cf_op": cf_op, "family": fam}
    q_true, q_tgt = build_clean_check_probes(subj, obj, rt, rg)
    q_open = "Where is the %s relative to the %s?" % (subj, obj)
    _contrast = bool(getattr(CFG, "CF_CONTRAST", False))
    for mi, m in enumerate(models):
        pv_c = to_pixel_values(clean01, m["mean"], m["std"], m["dtype"])
        pv_m = (to_pixel_values(x_m, m["mean"], m["std"], m["dtype"])
                if cf_op == "mirror" else pv_c)
        with torch.no_grad():
            if cf_op == "mirror":
                a_true, a_tgt = generate_batch(m, pil_m, [q_true, q_tgt], pv_m, max_new_tokens=6)
                a_open = generate(m, pil_m, q_open, pv_m)
            else:
                # Reuse clean gate check if cached on m, otherwise batch decode
                cg = m.get("_clean_gate_cache")
                if cg and cg.get("q_true") == q_true and cg.get("q_tgt") == q_tgt:
                    a_tgt, a_true = cg["a_true"], cg["a_tgt"]
                else:
                    a_tgt, a_true = generate_batch(m, image_pil, [q_true, q_tgt], pv_c, max_new_tokens=6)
                if fam in ("lateral", "vertical", "depth"):
                    q_open_cf = "Where is the %s relative to the %s?" % (obj, subj)
                    a_open = generate(m, image_pil, q_open_cf, pv_c)
                else:
                    a_open = generate(m, image_pil, q_open, pv_c)
        gate = (yesno(a_tgt) == "yes") and (yesno(a_true) == "no")
        if _contrast:
            with torch.no_grad():
                g_ct, g_cr = yesno_gap_batch(m, image_pil, pv_c, [q_tgt, q_true])
                if cf_op == "mirror":
                    g_mt, g_mr = yesno_gap_batch(m, pil_m, pv_m, [q_tgt, q_true])
                else:
                    g_mt, g_mr = g_cr, g_ct
            sw_t, sw_r = round(g_mt - g_ct, 3), round(g_cr - g_mr, 3)
            print("    [cct] (%s) yes/no swing: target-Q %+.2f (cf %+.2f, clean %+.2f) | "
                  "true-Q %+.2f (clean %+.2f, cf %+.2f)"
                  % (cf_op, sw_t, g_mt, g_ct, sw_r, g_cr, g_mr))
            if mi == 0:
                info.update(contrast=True, contrast_yn_swing_target=sw_t,
                            contrast_yn_swing_true=sw_r,
                            contrast_alpha=float(CFG.CF_CONTRAST_ALPHA),
                            contrast_js=bool(CFG.CF_CONTRAST_JS))
        open_ok = (bool(score_lie(a_open, "open", rt, rg, subj=subj, obj=obj))
                   if cf_op == "mirror"
                   else bool(score_lie(a_open, "open", rt, rg, subj=obj, obj=subj)))
        print("    [cf] (%s) teacher check: target-Q -> %r | true-Q -> %r | gate=%s"
              % (cf_op, a_tgt, a_true, gate))
        print("    [cf] (%s) open answer: %r  (states target: %s)"
              % (cf_op, a_open[:80], open_ok))
        if mi == 0:
            info.update(gate_yesno=gate, gate_open=open_ok, mirror_target_q=a_tgt,
                        mirror_true_q=a_true, mirror_open=a_open)
        yesno_on = bool(gate or not CFG.CF_REQUIRE_GATE)
        if (_contrast and getattr(CFG, "CF_CONTRAST_REPLACE_GATE", True)
                and getattr(CFG, "CF_CONTRAST_JS", True)):
            if not yesno_on:
                print("    [cct] hard yes/no gate failed, but CCT replaces it: "
                      "teacher stays ON and self-silences via JS weights")
            yesno_on = True
        if mi == 0:
            info["yesno_teacher_on"] = yesno_on
        if not yesno_on:
            print("    [cf] yes/no teacher gate FAILED -> yes/no rows use classic loss "
                  "(open-dir teacher evaluated independently below)")
        dual_yn_map = (_build_dual_yn_map(m, image_pil, subj, obj, rt, rg)
                       if cf_op == "dual" else None)
        m["_cf"] = {"on": True, "yesno_on": yesno_on, "mode": mode, "cf_op": cf_op,
                    "w": float(CFG.CF_WEIGHT),
                    "temp": float(CFG.CF_TEMP), "pv": pv_m, "cache": {}, "dcache": {},
                    "dir_ids": None, "dual_yn_map": dual_yn_map, "dual_open_tD": None,
                    "contrast": _contrast, "pv_clean": pv_c,
                    "alpha": float(getattr(CFG, "CF_CONTRAST_ALPHA", 1.0)),
                    "beta": float(getattr(CFG, "CF_CONTRAST_BETA", 0.1)),
                    "js": bool(getattr(CFG, "CF_CONTRAST_JS", True)),
                    "js_ref": float(getattr(CFG, "CF_CONTRAST_JS_REF", 0.15)),
                    "stats": {"teacher_rows": 0, "kl_sum": 0.0, "kl_n": 0, "calls": 0,
                              "dir_sum": 0.0, "dir_n": 0, "w_sum": 0.0, "w_n": 0}}
        # v18.11 MBPD LOSS: pre-attack per-frame table (train frames only).
        if getattr(CFG, "MBPD_LOSS", False):
            _tab = _mbpd_table(m, image_pil, pv_c, pv_m, subj, obj, rt, rg, cf_op=cf_op)
            m["_cf"]["mbpd"] = _tab
            if not _contrast:
                print("    [mbpd] WARNING: CF_CONTRAST=False -> per-frame alpha unused; "
                      "only the teacher switch and fragility weights apply")
            _cnt = {}
            for _e in _tab.values():
                _cnt[_e["status"]] = _cnt.get(_e["status"], 0) + 1
            _nw = [_e["w_mult"] for _k, _e in _tab.items() if _k.startswith("n")]
            print("    [mbpd] (%s) LOSS table: %s | no-row weight x%.2f..x%.2f | teacher on %d/%d rows"
                  % (cf_op, json.dumps(_cnt), min(_nw), max(_nw),
                     sum(_e["teacher"] for _e in _tab.values()), len(_tab)))
            for _k in sorted(_tab, key=lambda k: (int(k[1:]), k[0])):
                _e = _tab[_k]
                if _k.startswith("n"):
                    print("    [mbpd]   %-4s P=%+.2f g_cf=%+.2f %-19s alpha_loss=%-5s w=x%.2f"
                          % (_k, _e["P"], _e["g_mirror"], _e["status"],
                             "-" if _e["alpha_loss"] is None else "%.2f" % _e["alpha_loss"],
                             _e["w_mult"]))
            if mi == 0:
                info["mbpd_loss_table"] = _tab
                info["mbpd_loss_counts"] = _cnt
        open_dir_on = False
        if getattr(CFG, "CF_OPEN_DIR", False):
            tok = m["processor"].tokenizer
            _, _, i_true, i_tgt = _open_dir_slot_spec(tok, subj, rt, rg)
            ids2 = [i_true, i_tgt]
            _min_mass = float(getattr(CFG, "CF_OPEN_DIR_MIN_MASS", 0.0))
            ds_c = lateral_slot_stats(m, image_pil, pv_c, q_open, subj, rt, rg)
            open_cf_source = "mirror"
            if cf_op == "mirror":
                ds = lateral_slot_stats(m, pil_m, pv_m, q_open, subj, rt, rg)
            else:
                ds_swap = None
                if fam in ("lateral", "vertical", "depth"):
                    q_open_swap = "Where is the %s relative to the %s?" % (obj, subj)
                    ds_swap = lateral_slot_stats(m, image_pil, pv_c, q_open_swap, obj, rt, rg)
                if (ds_swap is not None and ds_swap["dir_gap"] > 0
                        and ds_swap.get("dir_mass", 0.0) >= _min_mass):
                    ds = ds_swap
                    open_cf_source = "converse_swap"
                else:
                    ds = dict(ds_c)
                    ds["dir_gap"] = round(-float(ds_c["dir_gap"]), 3)
                    open_cf_source = "polarity_dual"
                if ds["dir_gap"] >= 1.0 and ds.get("dir_mass", 0.0) < _min_mass:
                    ds["raw_dir_mass"] = ds.get("dir_mass", 0.0)
                    ds["dir_mass"] = round(min(0.95, float(ds["dir_gap"]) / 3.0), 4)
                _dg_t = float(ds["dir_gap"])
                _p_tgt = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, _dg_t))))
                _p_tru = max(1e-6, 1.0 - _p_tgt)
                _p_tgt = max(1e-6, _p_tgt)
                m["_cf"]["dual_open_tD"] = torch.tensor(
                    [[math.log(_p_tru), math.log(_p_tgt)]], device=device, dtype=torch.float32)
            print("    [cf] open-dir teacher (%s): cf dir_gap=%+.2f (>0 = prefers target) "
                  "dir_mass=%.3f (>=%.2f req) tokens=%r"
                  % (open_cf_source, ds["dir_gap"], ds.get("dir_mass", 0.0), _min_mass,
                     [tok.decode([i]) for i in ids2]))
            if mi == 0:
                info["mirror_dir_gap"] = ds["dir_gap"]
                info["mirror_dir_mass"] = ds.get("dir_mass", 0.0)
                info["open_cf_source"] = open_cf_source
            gap_ok = (ds["dir_gap"] > 0)
            mass_ok = (ds.get("dir_mass", 0.0) >= _min_mass)
            _swing = None
            if _contrast and getattr(CFG, "CF_CONTRAST_OPEN", True):
                _swing = float(ds["dir_gap"]) - float(ds_c["dir_gap"])
                print("    [cct] open swing (%s): dir_gap cf %+.2f - clean %+.2f = %+.2f"
                      % (open_cf_source, ds["dir_gap"], ds_c["dir_gap"], _swing))
                if mi == 0:
                    info["contrast_open_swing"] = round(_swing, 3)
                    info["clean_dir_gap"] = ds_c["dir_gap"]
                gap_ok = gap_ok and (_swing > 0)
            if CFG.CF_REQUIRE_GATE and not (gap_ok and mass_ok):
                why = ("cf teacher does not prefer target" if not gap_ok
                       else "cf teacher has no clear direction percept (dir_mass<%.2f)" % _min_mass)
                print("    [cf] open-dir gate FAILED (%s) -> open rows classic" % why)
            else:
                m["_cf"]["dir_ids"] = torch.tensor(ids2, device=device)
                open_dir_on = True
                _dm = float(ds.get("dir_mass", 0.0)); _dg = float(ds["dir_gap"])
                _c0 = float(getattr(CFG, "CF_PGCM_MASS_C0", 0.30))
                _sc = max(1e-3, float(getattr(CFG, "CF_PGCM_MASS_S", 0.10)))
                _sg = max(1e-3, float(getattr(CFG, "CF_PGCM_GAP_S", 1.50)))
                _mm = float(getattr(CFG, "CF_PGCM_MARGIN_MAX", 4.0))
                _pgcm = bool(getattr(CFG, "CF_PGCM", True))
                _beta = ((1.0 / (1.0 + math.exp(-(_dm - _c0) / _sc)))
                         * (1.0 / (1.0 + math.exp(-_dg / _sg)))) if _pgcm else 1.0
                if _swing is not None:
                    _s0 = float(getattr(CFG, "CF_CONTRAST_SWING_C0", 1.0))
                    _ss = max(1e-3, float(getattr(CFG, "CF_CONTRAST_SWING_S", 0.75)))
                    _beta *= 1.0 / (1.0 + math.exp(-(_swing - _s0) / _ss))
                _m0 = max(0.0, min(_dg, _mm))
                m["_cf"]["cf_beta"] = float(_beta)
                m["_cf"]["cf_m0"] = float(_m0)
                m["_cf"]["pgcm"] = _pgcm
                if _pgcm:
                    print("    [cf] PGCM (%s): beta=%.3f (percept trust) m0=%.2f "
                          "(teacher-anchored target margin, cap=%.1f)"
                          % (open_cf_source, _beta, _m0, _mm))
                if mi == 0:
                    info["open_dir_on"] = True
                    info["pgcm"] = _pgcm
                    info["cf_beta"] = round(float(_beta), 4)
                    info["cf_m0"] = round(float(_m0), 4)
        if not yesno_on and not open_dir_on:
            m["_cf"] = None
            if mi == 0:
                info["on"] = False
                info["reason"] = "teacher_gate_failed"
            print("    [cf] no CF teacher active -> classic loss for this example")
            continue
        if mi == 0:
            info["on"] = True
        print("    [cf] ON mode=%s cf_op=%s weight=%.2f temp=%.2f yesno=%s open_dir=%s"
              % (mode, cf_op, CFG.CF_WEIGHT, CFG.CF_TEMP, yesno_on, open_dir_on))
        _prewarm_cf_teacher_cache(m)
    return info


@torch.no_grad()
def _prewarm_cf_teacher_cache(m):
    """Pre-fill the teacher cache for all 48 base train probes in batched
    no-grad passes before APGD step 0, eliminating mid-loop teacher stalls."""
    cf = m.get("_cf")
    if not cf or not cf.get("on"):
        return
    probes = m.get("train_pool") or []
    if not probes:
        return
    max_r = max(12, int(getattr(CFG, "MAX_ROWS_PER_FWD", 28)))
    for chunk in _group_probes_by_rows(probes, max_r):
        rows, row_probe = [], []
        for pi, pr in enumerate(chunk):
            for _r in pr["refs"]:
                rows.append((_r[0], _r[2], _r[3], _r[4], _r[5], _r[6] if len(_r) > 6 else None))
                row_probe.append(pi)
        B = len(rows)
        Lmax = max(r[0].shape[1] for r in rows)
        padded = torch.full((B, Lmax), m["pad_id"], dtype=torch.long, device=device)
        attn   = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            ids = r[0]; L = ids.shape[1]; padded[i, -L:] = ids[0]; attn[i, -L:] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        if cf.get("yesno_on", cf.get("on")):
            _kinds = set(getattr(CFG, "CF_KINDS", ["yes", "no"]) or [])
            _mt = cf.get("mbpd") or {}
            def _teach(pk):
                e = _mt.get(pk)
                return True if e is None else bool(e["teacher"])
            keys = [(_cf_key(r[0], r[1])
                     if (chunk[row_probe[i]].get("kind") in _kinds
                         and _teach(chunk[row_probe[i]].get("pkey"))) else None)
                    for i, r in enumerate(rows)]
            pkeys = [chunk[row_probe[i]].get("pkey") for i in range(B)]
            _cf_fill_batched(m, cf, keys, padded, attn, pos, rows, pkeys=pkeys)
        if _cf_open_dir_on(cf):
            dkeys = [(_cf_key(r[0], r[1]) if chunk[row_probe[i]].get("kind") == "open" else None)
                     for i, r in enumerate(rows)]
            _cf_fill_dir_batched(m, cf, dkeys, padded, attn, pos, rows)


def _cf_report(m):
    cf = m.get("_cf")
    if not cf:
        return None
    s = cf["stats"]
    return {"teacher_rows": s["teacher_rows"], "loss_calls": s["calls"],
            "mean_kl": round(s["kl_sum"] / max(1, s["kl_n"]), 5),
            "mean_dir_kl": round(s.get("dir_sum", 0.0) / max(1, s.get("dir_n", 0)), 5),
            "open_dir_rows": s.get("dir_n", 0),
            # v18.9 CCT: mean per-position JS trust actually applied (0 = teacher
            # silent, 1 = full strength). None when the plain teacher is used.
            "contrast": bool(cf.get("contrast")),
            "mean_contrast_w": (round(s["w_sum"] / s["w_n"], 4)
                                if s.get("w_n") else None)}


def _loss_batched(m, pixel_values, probes, weight_map):
    """Weighted loss for a micro-batch. Returns (scalar_for_grad, per_probe_detached).
    Soft-min stays exact (all refs of each probe present); OPT-B weights are
    constants folded in per probe."""
    model = m["model"]
    rows, row_probe = [], []
    for pi, pr in enumerate(probes):
        for _r in pr["refs"]:
            # v18.8 (1a): refs carry a 7th element (per-row contrast margin).
            # Read it positionally with a fallback so a row minted by an older
            # code path still works.
            ids, _attn, alen, pmask = _r[0], _r[1], _r[2], _r[3]
            forbid_ids, evasion_ids = _r[4], _r[5]
            cmg = _r[6] if len(_r) > 6 else None
            rows.append((ids, alen, pmask, forbid_ids, evasion_ids, cmg))
            row_probe.append(pi)
    B = len(rows)
    Lmax = max(r[0].shape[1] for r in rows)
    padded = torch.full((B, Lmax), m["pad_id"], dtype=torch.long, device=device)
    attn   = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ids = r[0]; L = ids.shape[1]; padded[i, -L:] = ids[0]; attn[i, -L:] = 1
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)
    # Fill teacher caches BEFORE the student forward pass so teacher no_grad
    # forwards never overlap in VRAM with the student autograd computation graph.
    cf = m.get("_cf")
    cf_yn = bool(cf and cf.get("yesno_on", cf.get("on")))
    if cf_yn:
        _kinds = set(getattr(CFG, "CF_KINDS", ["yes", "no"]) or [])
        _mt = cf.get("mbpd") or {}
        def _teach(pk):
            e = _mt.get(pk)
            return True if e is None else bool(e["teacher"])
        keys = [(_cf_key(r[0], r[1])
                 if (probes[row_probe[i]].get("kind") in _kinds
                     and _teach(probes[row_probe[i]].get("pkey"))) else None)
                for i, r in enumerate(rows)]
        pkeys = [probes[row_probe[i]].get("pkey") for i in range(len(rows))]
        n_before = len(cf["cache"])
        _cf_fill_batched(m, cf, keys, padded, attn, position_ids, rows, pkeys=pkeys)
        n_miss = len(cf["cache"]) - n_before
        kl_acc = torch.zeros((), device=device)
    _odir = _cf_open_dir_on(cf)
    if _odir:
        dkeys = [(_cf_key(r[0], r[1]) if (probes[row_probe[i]].get("kind") == "open"
                                          and not probes[row_probe[i]].get("converse", False)) else None)
                 for i, r in enumerate(rows)]
        n_miss_d = _cf_fill_dir_batched(m, cf, dkeys, padded, attn, position_ids, rows)
        dir_acc = torch.zeros((), device=device)
    max_alen = max(int(r[1]) for r in rows)
    rc = m.get("repcon_cfg")
    _keep = None if (rc is not None and rc.enable) else (max_alen + 2)
    out = _model_fwd_tail(model, input_ids=padded, attention_mask=attn,
                          position_ids=position_ids,
                          pixel_values=pixel_values.expand(B, -1, -1, -1),
                          logits_to_keep=_keep)
    logits = out.logits
    per_row = []
    for i, (_ids, alen, pmask, forbid_ids, evasion_ids, cmg) in enumerate(rows):
        sl = logits[i, -alen - 1:-1, :]; lbl = padded[i, -alen:]
        if cf_yn and keys[i] is not None:
            rl, kl_d = _cf_row(m, cf, sl, lbl, rows[i], cf["cache"][keys[i]])
            kl_acc = kl_acc + kl_d
            per_row.append(rl)
        elif _odir and dkeys[i] is not None:
            dl = _cf_dir_loss(sl, cf["dcache"][dkeys[i]], cf["dir_ids"],
                              beta=cf.get("cf_beta", 1.0), m0=cf.get("cf_m0", 0.0),
                              pgcm=cf.get("pgcm", True))
            dir_acc = dir_acc + dl.detach()
            per_row.append(_row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg)
                           + float(CFG.CF_OPEN_WEIGHT) * dl)
        else:
            per_row.append(_row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg))
    if cf_yn or _odir:
        _cf_debug(cf,
                  kl_acc if cf_yn else 0.0,
                  sum(1 for k in keys if k is not None) if cf_yn else 0,
                  n_miss if cf_yn else 0,
                  dir_acc if _odir else 0.0,
                  sum(1 for k in dkeys if k is not None) if _odir else 0)
    total = torch.zeros((), device=device); detached = []
    no_losses = []
    for pi, pr in enumerate(probes):
        idxs = [i for i in range(B) if row_probe[i] == pi]
        lref = torch.stack([per_row[i] for i in idxs])
        Lp = _softmin(lref, CFG.SOFTMIN_TAU)
        w = weight_map.get(pr["pkey"], CFG.LAMBDA_CONSIST) * _mbpd_wmult(m, pr)
        total = total + w * Lp
        if pr.get("kind") == "no":
            no_losses.append(Lp)
        detached.append((pr["pkey"], float(Lp.detach())))
    total = total + _polarity_invariance(m, no_losses)
    # IDEA-C: representation-space belief consistency (batched path). The tap
    # captured this chunk's residual stream during model(...) above; rows[i] is
    # (ids, alen, pmask, forbid_ids, evasion_ids, cmg) so [1]=alen, [2]=pmask.
    total = total + _repcon_add(m, rows, B)
    return total, detached

def _repcon_add(m, rows, B):
    """Add the repcon term for a batched chunk. Returns 0.0 if disabled/inert.
    Kept here so both _loss_batched and _loss_sequential stay small."""
    rc = m.get("repcon_cfg")
    if rc is None or not rc.enable or m.get("_repdir") is None or not m["_repdir"].ok:
        return 0.0
    tap = m.get("_reptap")
    if tap is None or tap.act is None:
        return 0.0
    import repcon
    rows_meta = [(i, rows[i][1], rows[i][2], rows[i][0]) for i in range(B)]  # (i,alen,pmask,ids)
    tok = m["processor"].tokenizer
    rloss, rtele = repcon.repcon_loss(tap.act, rows_meta, m["_repdir"], rc, tok)
    if rtele:
        m["_rep_last"] = rtele
    return rloss

def _polarity_invariance(m, no_losses):
    """v18.8 (1c). FRAME-INVARIANCE PENALTY on the no-family.

    The measured failure is frame-specific assent, not a low mean: "Can you
    confirm that X" drew "Yes." on BOTH polarities while "Would it be accurate
    to say X" answered correctly on the same image. A weighted SUM over probes
    is indifferent between {2.0, 2.0} and {0.5, 3.5} at equal mean, so raising
    the mean no-probe margin over the train frames has no reason to equalise
    them -- and it demonstrably did not transfer to the held-out frame.
    Penalising the SPREAD does have that reason.

    SCOPE CAVEAT, stated because it bounds what this can do: the micro-batched
    accumulation in loss_and_grad builds one graph per CHUNK, so the variance is
    taken over the no-probes present in THIS chunk, not over all of them in the
    step. With DRO_SUBSET=8 the stratifier yields ~2 no-probes and
    MAX_ROWS_PER_FWD=8 may split them, so the term can fire on fewer pairs than
    expected. `polarity_invar_fires` counts how often it actually ran -- read it
    before concluding the mechanism does or does not work. If it is near zero,
    raise DRO_SUBSET or MAX_ROWS_PER_FWD rather than raising the weight.
    """
    if not CFG.POLARITY_INVAR or len(no_losses) < 2:
        if CFG.POLARITY_INVAR:
            m["_polinv_skips"] = int(m.get("_polinv_skips", 0)) + 1
        return 0.0
    v = torch.stack(no_losses).var(unbiased=False)
    m["_polinv_fires"] = int(m.get("_polinv_fires", 0)) + 1
    m["_polinv_last"] = float(v.detach())
    return float(CFG.POLARITY_INVAR_W) * v

def _loss_sequential(m, pixel_values, probes, weight_map):
    total = torch.zeros((), device=device); detached = []
    no_losses = []
    rc = m.get("repcon_cfg")
    _rep_on = (rc is not None and rc.enable and m.get("_repdir") is not None
               and m["_repdir"].ok and m.get("_reptap") is not None)
    rep_H = []
    for pr in probes:
        lref = []
        for _r in pr["refs"]:
            ids, attn, alen, pmask = _r[0], _r[1], _r[2], _r[3]
            forbid_ids, evasion_ids = _r[4], _r[5]
            cmg = _r[6] if len(_r) > 6 else None
            out = m["model"](input_ids=ids, attention_mask=attn,
                             pixel_values=pixel_values, use_cache=False)
            if _rep_on and m["_reptap"].act is not None:
                import repcon
                _rc = m.get("repcon_cfg")
                _word = (_rc.obj_word if _rc.dir_target == "object"
                         else (_rc.subj_word if _rc.dir_target == "subject" else None))
                if _word:
                    _wm = repcon._relation_token_span(m["processor"].tokenizer, ids, alen, _word)
                    rep_H.append(repcon._decision_hidden(m["_reptap"].act[0], alen, _wm))
                else:
                    rep_H.append(repcon._decision_hidden(m["_reptap"].act[0], alen, pmask))
            sl = out.logits[0, -alen - 1:-1, :]; lbl = ids[0, -alen:]
            _cf = m.get("_cf")
            _mt = (_cf or {}).get("mbpd") or {}
            _te = _mt.get(pr.get("pkey"))
            if (_cf and _cf.get("yesno_on", _cf.get("on"))
                    and pr.get("kind") in set(getattr(CFG, "CF_KINDS", ["yes", "no"]) or [])
                    and (_te is None or _te["teacher"])):
                _k = _cf_key(ids, alen)
                _n0 = len(_cf["cache"])
                _cf_fill_single(m, _cf, _k, ids, attn, alen, pkey=pr.get("pkey"))
                rl, kl_d = _cf_row(m, _cf, sl, lbl,
                                   (ids, alen, pmask, forbid_ids, evasion_ids, cmg),
                                   _cf["cache"][_k])
                _cf_debug(_cf, kl_d, 1, len(_cf["cache"]) - _n0)
                lref.append(rl)
            elif _cf_open_dir_on(_cf) and pr.get("kind") == "open":
                _k = _cf_key(ids, alen)
                _cf_fill_dir_single(m, _cf, _k, ids, attn, alen, pmask=pmask)
                dl = _cf_dir_loss(sl, _cf["dcache"][_k], _cf["dir_ids"],
                                  beta=_cf.get("cf_beta", 1.0), m0=_cf.get("cf_m0", 0.0),
                                  pgcm=_cf.get("pgcm", True))
                _cf_debug(_cf, 0.0, 0, 0, dl.detach(), 1)
                lref.append(_row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg)
                            + float(CFG.CF_OPEN_WEIGHT) * dl)
            else:
                lref.append(_row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg))
        Lp = _softmin(torch.stack(lref), CFG.SOFTMIN_TAU)
        w = weight_map.get(pr["pkey"], CFG.LAMBDA_CONSIST) * _mbpd_wmult(m, pr)
        total = total + w * Lp
        if pr.get("kind") == "no":
            no_losses.append(Lp)
        detached.append((pr["pkey"], float(Lp.detach())))
    total = total + _polarity_invariance(m, no_losses)
    if _rep_on and rep_H:
        import repcon
        rloss, rtele = repcon.repcon_core(torch.stack(rep_H), m["_repdir"], rc)
        total = total + rloss
        if rtele:
            m["_rep_last"] = rtele
    return total, detached


# =============================================================================
# 6c. v12 memory-safe loss + gradient via micro-batched accumulation,
#     + OPT4 antithetic/family EOT, + OPT-B detached hard-probe weighting.
# =============================================================================
def _group_probes_by_rows(probes, max_rows):
    chunks, cur, cur_rows = [], [], 0
    for pr in probes:
        r = max(1, len(pr["refs"]))
        if cur and cur_rows + r > max_rows:
            chunks.append(cur); cur, cur_rows = [], 0
        cur.append(pr); cur_rows += r
    if cur:
        chunks.append(cur)
    return chunks

def _loss_batched_with_preserve(m, pixel_values, probes, weight_map, pres_rows):
    """Fused forward pass: runs all probe rows AND the 2 preservation rows in ONE
    single model() forward+backward pass, eliminating 100+ redundant 13B passes
    per image with bit-identical gradients."""
    if not pres_rows:
        Lc, det = _loss_batched(m, pixel_values, probes, weight_map)
        return Lc, det, None
    model = m["model"]
    rows, row_probe = [], []
    for pi, pr in enumerate(probes):
        for _r in pr["refs"]:
            ids, _attn, alen, pmask = _r[0], _r[1], _r[2], _r[3]
            forbid_ids, evasion_ids = _r[4], _r[5]
            cmg = _r[6] if len(_r) > 6 else None
            rows.append((ids, alen, pmask, forbid_ids, evasion_ids, cmg))
            row_probe.append(pi)
    B_probes = len(rows)
    B_pres = len(pres_rows)
    B_tot = B_probes + B_pres
    Lmax = max([r[0].shape[1] for r in rows] + [r["ids"].shape[1] for r in pres_rows])
    padded = torch.full((B_tot, Lmax), m["pad_id"], dtype=torch.long, device=device)
    attn   = torch.zeros((B_tot, Lmax), dtype=torch.long, device=device)
    for i, r in enumerate(rows):
        ids = r[0]; L = ids.shape[1]; padded[i, -L:] = ids[0]; attn[i, -L:] = 1
    for j, r in enumerate(pres_rows):
        ids = r["ids"]; L = ids.shape[1]; padded[B_probes + j, -L:] = ids[0]; attn[B_probes + j, -L:] = 1
    position_ids = (attn.cumsum(-1) - 1).clamp(min=0)

    cf = m.get("_cf")
    cf_yn = bool(cf and cf.get("yesno_on", cf.get("on")))
    if cf_yn:
        _kinds = set(getattr(CFG, "CF_KINDS", ["yes", "no"]) or [])
        _mt = cf.get("mbpd") or {}
        def _teach(pk):
            e = _mt.get(pk)
            return True if e is None else bool(e["teacher"])
        keys = [(_cf_key(r[0], r[1])
                 if (probes[row_probe[i]].get("kind") in _kinds
                     and _teach(probes[row_probe[i]].get("pkey"))) else None)
                for i, r in enumerate(rows)]
        pkeys = [probes[row_probe[i]].get("pkey") for i in range(B_probes)]
        n_before = len(cf["cache"])
        _cf_fill_batched(m, cf, keys, padded[:B_probes], attn[:B_probes], position_ids[:B_probes], rows, pkeys=pkeys)
        n_miss = len(cf["cache"]) - n_before
        kl_acc = torch.zeros((), device=device)
    _odir = _cf_open_dir_on(cf)
    if _odir:
        dkeys = [(_cf_key(r[0], r[1]) if (probes[row_probe[i]].get("kind") == "open"
                                          and not probes[row_probe[i]].get("converse", False)) else None)
                 for i, r in enumerate(rows)]
        n_miss_d = _cf_fill_dir_batched(m, cf, dkeys, padded[:B_probes], attn[:B_probes], position_ids[:B_probes], rows)
        dir_acc = torch.zeros((), device=device)

    max_alen = max([int(r[1]) for r in rows] + [int(r["alen"]) for r in pres_rows])
    rc = m.get("repcon_cfg")
    _keep = None if (rc is not None and rc.enable) else (max_alen + 2)
    out = _model_fwd_tail(model, input_ids=padded, attention_mask=attn,
                          position_ids=position_ids,
                          pixel_values=pixel_values.expand(B_tot, -1, -1, -1),
                          logits_to_keep=_keep)
    logits = out.logits
    per_row = []
    for i, (_ids, alen, pmask, forbid_ids, evasion_ids, cmg) in enumerate(rows):
        sl = logits[i, -alen - 1:-1, :]; lbl = padded[i, -alen:]
        if cf_yn and keys[i] is not None:
            rl, kl_d = _cf_row(m, cf, sl, lbl, rows[i], cf["cache"][keys[i]])
            kl_acc = kl_acc + kl_d
            per_row.append(rl)
        elif _odir and dkeys[i] is not None:
            dl = _cf_dir_loss(sl, cf["dcache"][dkeys[i]], cf["dir_ids"],
                              beta=cf.get("cf_beta", 1.0), m0=cf.get("cf_m0", 0.0),
                              pgcm=cf.get("pgcm", True))
            dir_acc = dir_acc + dl.detach()
            per_row.append(_row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg)
                           + float(CFG.CF_OPEN_WEIGHT) * dl)
        else:
            per_row.append(_row_loss(sl, lbl, pmask, forbid_ids, evasion_ids, cmg))
    if cf_yn or _odir:
        _cf_debug(cf,
                  kl_acc if cf_yn else 0.0,
                  sum(1 for k in keys if k is not None) if cf_yn else 0,
                  n_miss if cf_yn else 0,
                  dir_acc if _odir else 0.0,
                  sum(1 for k in dkeys if k is not None) if _odir else 0)
    total = torch.zeros((), device=device); detached = []
    no_losses = []
    for pi, pr in enumerate(probes):
        idxs = [i for i in range(B_probes) if row_probe[i] == pi]
        lref = torch.stack([per_row[i] for i in idxs])
        Lp = _softmin(lref, CFG.SOFTMIN_TAU)
        w = weight_map.get(pr["pkey"], CFG.LAMBDA_CONSIST) * _mbpd_wmult(m, pr)
        total = total + w * Lp
        if pr.get("kind") == "no":
            no_losses.append(Lp)
        detached.append((pr["pkey"], float(Lp.detach())))
    total = total + _polarity_invariance(m, no_losses)
    total = total + _repcon_add(m, rows, B_probes)

    pres_tot = torch.zeros((), device=device)
    for j, r in enumerate(pres_rows):
        alen = r["alen"]
        sl = logits[B_probes + j, -alen - 1:-1, :].float()
        lbl = padded[B_probes + j, -alen:]
        pres_tot = pres_tot + F.cross_entropy(sl, lbl)
    Lpres = pres_tot / max(1, B_pres)
    return total, detached, Lpres


def loss_and_grad(models, x, prompt_eot, eot_samples=None, progress=0.0,
                  need_grad=True):
    """Returns (loss_detached, image_grad_detached).
    When `need_grad=False` (used by APGD's `f_best` refreshes after on-policy
    minting or dual updates), runs with 1 EOT draw under `torch.no_grad()`,
    and fuses `preserve_pool` rows into the first chunk so preservation never
    triggers a separate 13B forward+backward pass."""
    xr = x.detach().clone().requires_grad_(bool(need_grad))
    T_full = max(1, eot_samples if eot_samples is not None else CFG.EOT_SAMPLES)
    if not need_grad:
        T = 1
    elif getattr(CFG, "TWO_STAGE_EOT", True) and float(progress) < float(getattr(CFG, "EOT_STAGE2_FRAC", 0.70)):
        T = 1
    else:
        T = T_full
    antithetic = (getattr(CFG, "EOT_ANTITHETIC", False)
                  and CFG.EOT_DITHER > 0 and T >= 2)
    max_rows = max(1, getattr(CFG, "MAX_ROWS_PER_FWD", 28))
    denom = float(T * len(models))
    total_loss = torch.zeros((), device=device)
    total_grad = torch.zeros_like(xr)
    grad_ctx = torch.enable_grad if need_grad else torch.no_grad
    with grad_ctx():
        for m in models:
            probes = _active_probes(m, prompt_eot)
            weight_map = _probe_weights(m, probes, progress)
            probes = [p for p in probes if weight_map.get(p["pkey"], 0.0) > 0.0]
            dithers = _make_dithers(xr, T, antithetic)
            pres_rows = (m.get("preserve_pool") or []) if CFG.PRESERVE_LOSS else []
            lam = float(m.get("preserve_lambda", CFG.PRESERVE_WEIGHT)) if pres_rows else 0.0
            pres_done = False
            if probes:
                chunks = (_group_probes_by_rows(probes, max_rows) if CFG.BATCH_PROBES
                          else [[p] for p in probes])
                accum = {}
                for t in range(T):
                    dith = dithers[t]
                    pending = list(chunks)
                    while pending:
                        chunk = pending.pop(0)
                        fuse_pres = (CFG.BATCH_PROBES and (not pres_done) and bool(pres_rows))
                        try:
                            xt = eot_view(xr, dith)
                            pv = to_pixel_values(xt, m["mean"], m["std"], m["dtype"])
                            if fuse_pres:
                                Lc, det, Lpres = _loss_batched_with_preserve(
                                    m, pv, chunk, weight_map, pres_rows)
                                m["_preserve_last"] = float(Lpres.detach())
                                Lterm = (Lc / denom) + (
                                    (lam * Lpres) / float(max(1, len(models)))
                                    if lam > 0.0 else 0.0
                                )
                                pres_done = True
                            else:
                                Lc, det = (_loss_batched(m, pv, chunk, weight_map) if CFG.BATCH_PROBES
                                           else _loss_sequential(m, pv, chunk, weight_map))
                                Lterm = Lc / denom
                            if need_grad:
                                g, = torch.autograd.grad(Lterm, xr)
                                total_grad = total_grad + g.detach()
                                del g
                            total_loss = total_loss + Lterm.detach()
                            for pkey, val in det:
                                accum.setdefault(pkey, []).append(val)
                            del xt, pv, Lc, Lterm
                        except torch.cuda.OutOfMemoryError:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if len(chunk) > 1:
                                mid = max(1, len(chunk) // 2)
                                CFG.MAX_ROWS_PER_FWD = max(8, min(int(CFG.MAX_ROWS_PER_FWD), mid * 2))
                                print(f"    [mem] OOM on chunk of {len(chunk)} probes; splitting & clamping MAX_ROWS_PER_FWD={CFG.MAX_ROWS_PER_FWD}")
                                pending = [chunk[:mid], chunk[mid:]] + pending
                            else:
                                raise
                _update_probe_ema(m, accum)
                if CFG.PROBE_LAGRANGIAN and _CRN.get("phase", "main") == "main":
                    _cur = {k: (sum(v) / max(1, len(v))) for k, v in accum.items()}
                    m["_probe_last"] = _cur
                    _seen = m.setdefault("_probe_last_seen", {})
                    _step = int(m.get("_probe_dual_steps", 0))
                    for _k, _v in _cur.items():
                        _seen[_k] = (float(_v), _step)

            if CFG.PRESERVE_LOSS and pres_rows and (not pres_done):
                dith0 = dithers[0] if dithers else None
                if lam > 0.0 and need_grad:
                    xt = eot_view(xr, dith0)
                    pv = to_pixel_values(xt, m["mean"], m["std"], m["dtype"])
                    Lpres = _preserve_loss(m, pv, pres_rows)
                    m["_preserve_last"] = float(Lpres.detach())
                    Lterm = (lam * Lpres) / float(max(1, len(models)))
                    g, = torch.autograd.grad(Lterm, xr)
                    total_grad = total_grad + g.detach()
                    total_loss = total_loss + Lterm.detach()
                    del xt, pv, Lpres, Lterm, g
                else:
                    with torch.no_grad():
                        xt = eot_view(xr, dith0)
                        pv = to_pixel_values(xt, m["mean"], m["std"], m["dtype"])
                        Lpres = _preserve_loss(m, pv, pres_rows)
                        m["_preserve_last"] = float(Lpres.detach())
                        if lam > 0.0:
                            total_loss = total_loss + (lam * Lpres.detach()) / float(max(1, len(models)))
                        del xt, pv, Lpres
    return total_loss.detach(), total_grad.detach()


# =============================================================================
# 7. Generation (eval only)
# =============================================================================
@torch.no_grad()
def generate(m, image_pil, question, pixel_values, max_new_tokens=None):
    prompt = PROMPT_TMPL.format(q=question)
    enc = _fast_encode_text(m["processor"], image_pil, prompt)
    input_ids = enc["input_ids"].to(device); attn = enc["attention_mask"].to(device)
    gen = m["model"].generate(input_ids=input_ids, attention_mask=attn,
                              pixel_values=pixel_values,
                              max_new_tokens=int(max_new_tokens or CFG.GEN_MAX_NEW_TOKENS),
                              do_sample=False,
                              use_cache=True)
    return m["processor"].tokenizer.decode(gen[0, input_ids.shape[1]:],
                                           skip_special_tokens=True).strip()


@torch.no_grad()
def generate_batch(m, image_pil, questions, pixel_values, max_new_tokens=None):
    """Batched greedy generation with left-padding across multiple questions on
    the same image. Combined with _install_vision_dedup, runs the CLIP vision
    tower ONCE (`batch_size=1`) and decodes all B prompts in parallel."""
    if not questions:
        return []
    if len(questions) == 1 or not getattr(CFG, "BATCH_GENERATE", True):
        return [generate(m, image_pil, q, pixel_values, max_new_tokens=max_new_tokens)
                for q in questions]
    B = len(questions)
    encs = [_fast_encode_text(m["processor"], image_pil, PROMPT_TMPL.format(q=q))
            for q in questions]
    Lmax = max(int(e["input_ids"].shape[1]) for e in encs)
    padded = torch.full((B, Lmax), m["pad_id"], dtype=torch.long, device=device)
    attn   = torch.zeros((B, Lmax), dtype=torch.long, device=device)
    for i, e in enumerate(encs):
        ids = e["input_ids"][0]
        L = ids.shape[0]
        padded[i, -L:] = ids.to(device)
        attn[i, -L:] = 1
    gen = m["model"].generate(
        input_ids=padded,
        attention_mask=attn,
        pixel_values=pixel_values.expand(B, -1, -1, -1),
        max_new_tokens=int(max_new_tokens or CFG.GEN_MAX_NEW_TOKENS),
        do_sample=False,
        use_cache=True,
        pad_token_id=m["pad_id"],
    )
    tok = m["processor"].tokenizer
    return [tok.decode(gen[i, Lmax:], skip_special_tokens=True).strip()
            for i in range(B)]


# =============================================================================
# 7b. IDEA-2: on-policy prefix augmentation (scheduled sampling for attacks)
# =============================================================================
@torch.no_grad()
def _onpolicy_update(models, x_adv, step, n_iter):
    """Greedily decode the CURRENT adversarial image (batched across open and
    yes/no rollout probes). For every OPEN probe that still fails score_lie(),
    mint a training row whose context is the model's own wrong prefix and whose
    labels are the target continuation."""
    changed = False
    for m in models:
        ctx = m.get("_op_ctx")
        if ctx is None or not m.get("train_pool"):
            continue
        rt, rg, img = ctx["rt"], ctx["rg"], ctx["image_pil"]
        subj, obj = ctx["subj"], ctx["obj"]
        stats = m.setdefault("_op_stats", {"triggers": 0, "rollouts": 0,
                                           "minted": 0, "boosted": 0,
                                           "open_fail": 0, "yn_fail": 0,
                                           "mint_skipped": 0, "mint_gated": 0,
                                           "gate_reasons": {}})
        stats["triggers"] += 1
        tok = m["processor"].tokenizer
        xq = quant_ste(x_adv.detach().clamp(0.0, 1.0))
        pv = to_pixel_values(xq, m["mean"], m["std"], m["dtype"])

        opens = [p for p in m["train_pool"] if p["kind"] == "open" and not p.get("converse", False)]
        yesno = [p for p in m["train_pool"] if p["kind"] in ("yes", "no") and not p.get("converse", False)]
        random.shuffle(opens); random.shuffle(yesno)

        minted, boosted = 0, 0
        _obs, _n_roll, _n_fail = [], 0, 0
        _fam = {}
        open_sel = opens[:max(0, CFG.ONPOLICY_OPEN_PROBES)]
        open_ans_list = generate_batch(
            m, img, [pr["q"] for pr in open_sel], pv,
            max_new_tokens=CFG.ONPOLICY_GEN_TOKENS)
        for pr, ans in zip(open_sel, open_ans_list):
            stats["rollouts"] += 1
            _n_roll += 1
            _ok = bool(ans) and bool(score_lie(ans, "open", rt, rg,
                                               subj=subj, obj=obj))
            _obs.append((m["probe_ema"].get(pr["pkey"]), 0.0 if _ok else 1.0))
            _f = _fam.setdefault("open", [0, 0]); _f[0] += 1; _f[1] += 0 if _ok else 1
            if not _ok:
                _n_fail += 1
            if not ans or _ok:
                continue
            stats["open_fail"] += 1

            if CFG.ONPOLICY_MINT_GATE:
                bad = None
                if CFG.MINT_REJECT_DEGEN and is_degenerate(ans, subj, obj):
                    bad = "degenerate"
                elif CFG.MINT_REQUIRE_SUBJECT and not _mentions(ans.lower(), subj):
                    bad = "subject_absent"
                elif CFG.MINT_REQUIRE_OBJECT and not _mentions(ans.lower(), obj):
                    bad = "object_absent"
                if bad:
                    stats["mint_gated"] += 1
                    stats.setdefault("gate_reasons", {})
                    stats["gate_reasons"][bad] = stats["gate_reasons"].get(bad, 0) + 1
                    vlog("    [onpol] gated (%s): %r" % (bad, ans[:70]), 2)
                    continue

            target = pr["answers"][0]
            ps = split_prefix_suffix(tok, pr["q"], ans, target,
                                     CFG.ONPOLICY_PREFIX_TOKENS,
                                     subj=subj, rg=rg)
            if ps is None:
                stats["mint_skipped"] += 1
                continue
            pref, suff = ps
            enc = encode_probe_prefixed(m["processor"], img, pr["q"], pref, suff)
            if enc is None:
                stats["mint_skipped"] += 1
                continue
            ids, attn, alen = enc
            grp = relation_group(rg)
            ans_ids = ids[0, -alen:].tolist()
            pm = (pivot_token_mask(tok, ans_ids, grp)
                  if (CFG.PIVOT_ONLY and grp is not None) else [True] * alen)
            pmask = torch.tensor(pm, dtype=torch.bool, device=device)
            fid = m["_ids"]["forbid"]; eid = m["_ids"]["evasion"]
            row = {"refs": [(ids, attn, alen, pmask, fid, eid,
                             float(CFG.CONTRAST_MARGIN))],
                   "kind": "open",
                   "pkey": "op%d_%s" % (step, str(pr["pkey"])),
                   "variant": 0, "q": pr["q"],
                   "answers": [suff], "prefix": pref}
            m["onpolicy_pool"].append(row)
            minted += 1; stats["minted"] += 1
            vlog("    [onpol] step %d mint <- %r || %r" % (step, pref.strip(),
                                                           suff.strip()), 2)

        # ---- IDEA-N3: Alternating Scene-Caption & 2-Turn Follow-Up Minting ----
        # Directly targets the two weakest held-out structural probes (`cap_eval`
        # and `fup_eval`) without flooding the 10-row single-turn FIFO buffer:
        #   - Steps 20, 60: rolls out a scene-caption prompt (disjoint from cap_eval)
        #   - Steps 40, 80: rolls out a 2-turn dialogue grounded on clean caption
        if getattr(CFG, "NOVEL_MULTITURN_MINT", False):
            if (step // max(1, int(CFG.ONPOLICY_EVERY))) % 2 == 1:
                mt_q = f"Provide a brief caption of this scene that states where the {subj} is in relation to the {obj}."
            else:
                pres_ans = ""
                if m.get("preserve_pool"):
                    pres_ans = str(m["preserve_pool"][0].get("answer", "")).strip()
                turn1_a = (pres_ans.split(".")[0].strip() + "."
                           if pres_ans else f"In this image, both the {subj} and the {obj} are visible.")
                mt_q = (f"Briefly describe the objects in the scene. ASSISTANT: {turn1_a} "
                        f"USER: Follow-up: where is the {subj} positioned relative to the {obj}?")
            mt_ans_list = generate_batch(m, img, [mt_q], pv,
                                         max_new_tokens=CFG.ONPOLICY_GEN_TOKENS)
            mt_ans = mt_ans_list[0] if mt_ans_list else ""
            stats["rollouts"] += 1
            if not (mt_ans and score_lie(mt_ans, "open", rt, rg, subj=subj, obj=obj)):
                stats["open_fail"] += 1
                target_mt = build_target_answers(subj, obj, rt, rg)[0]
                ps_mt = split_prefix_suffix(tok, mt_q, mt_ans or f"In this scene, the {subj} is",
                                            target_mt, CFG.ONPOLICY_PREFIX_TOKENS,
                                            subj=subj, rg=rg)
                if ps_mt is not None:
                    pref_mt, suff_mt = ps_mt
                    enc_mt = encode_probe_prefixed(m["processor"], img, mt_q, pref_mt, suff_mt)
                else:
                    pref_mt, suff_mt = f"In this scene, the {subj} is", f"{rg} the {obj}."
                    enc_mt = encode_probe_prefixed(m["processor"], img, mt_q, pref_mt, suff_mt)
                if enc_mt is not None:
                    ids_m, attn_m, alen_m = enc_mt
                    grp_m = relation_group(rg)
                    pm_m = (pivot_token_mask(tok, ids_m[0, -alen_m:].tolist(), grp_m)
                            if (CFG.PIVOT_ONLY and grp_m is not None) else [True] * alen_m)
                    pmask_m = torch.tensor(pm_m, dtype=torch.bool, device=device)
                    fid = m["_ids"]["forbid"]; eid = m["_ids"]["evasion"]
                    # Keep at most 2 multi-turn/caption rows in the buffer so single-turn rows aren't evicted
                    _mt_rows = [r for r in m["onpolicy_pool"] if str(r.get("pkey", "")).startswith("op_mt_")]
                    if len(_mt_rows) >= 2:
                        _drop_pk = _mt_rows[0]["pkey"]
                        m["onpolicy_pool"] = [r for r in m["onpolicy_pool"] if r.get("pkey") != _drop_pk]
                    m["onpolicy_pool"].append({
                        "refs": [(ids_m, attn_m, alen_m, pmask_m, fid, eid,
                                  float(CFG.CONTRAST_MARGIN))],
                        "kind": "open",
                        "pkey": "op_mt_%d" % step,
                        "variant": 0, "q": mt_q,
                        "answers": [suff_mt], "prefix": pref_mt,
                    })
                    minted += 1; stats["minted"] += 1
                    vlog("    [onpol-mt] step %d cap/2-turn mint <- %r || %r"
                         % (step, pref_mt.strip(), suff_mt.strip()), 2)

        cap = max(1, CFG.ONPOLICY_BUFFER)
        if len(m["onpolicy_pool"]) > cap:
            m["onpolicy_pool"] = m["onpolicy_pool"][-cap:]

        ema = m["probe_ema"]
        _finite = [v for v in ema.values() if math.isfinite(v)]
        mx = max(_finite) if _finite else 1.0
        yn_sel = yesno[:max(0, CFG.ONPOLICY_YESNO_PROBES)]
        yn_ans_list = generate_batch(
            m, img, [pr["q"] for pr in yn_sel], pv, max_new_tokens=6)
        for pr, ans in zip(yn_sel, yn_ans_list):
            stats["rollouts"] += 1
            _n_roll += 1
            _ok = bool(score_lie(ans, pr["kind"], rt, rg, subj=subj, obj=obj))
            _obs.append((ema.get(pr["pkey"]), 0.0 if _ok else 1.0))
            _f = _fam.setdefault(pr["kind"], [0, 0])
            _f[0] += 1; _f[1] += 0 if _ok else 1
            if _ok:
                continue
            _n_fail += 1
            stats["yn_fail"] += 1
            # v18.5: "observe" stops the EMA hijack. In "ema" mode this line
            # writes max(all EMAs)*1.25 into the probe, which puts it STRICTLY
            # above everything else and guarantees it a tail slot for many
            # steps (the EMA only decays when the probe is scored, and a
            # boosted probe is always selected). At alpha=0.35, k=3 -- so up to
            # two thirds of the gradient budget went to the family already at
            # sbc_yes = 1.000. The failure is still recorded above; it just no
            # longer overrides the ranking.
            if str(getattr(CFG, "ONPOLICY_BOOST_MODE", "ema")).lower() == "ema":
                ema[pr["pkey"]] = max(
                    (ema.get(pr["pkey"], 0.0)
                     if math.isfinite(ema.get(pr["pkey"], 0.0)) else 0.0),
                    mx * float(CFG.ONPOLICY_FAIL_BOOST))
                boosted += 1; stats["boosted"] += 1

        # ---- v18.5 (i): alpha_t <- EMA of the measured failure fraction ------
        if _n_roll > 0:
            f_now = float(_n_fail) / float(_n_roll)
            prev = m.get("_op_fail_frac")
            b = float(CFG.CVAR_ADAPT_BETA)
            m["_op_fail_frac"] = f_now if prev is None else (b * prev + (1.0 - b) * f_now)
            stats["fail_frac"] = round(float(m["_op_fail_frac"]), 4)
        # ---- v18.6 FMR: per-family failure-rate EMA --------------------------
        # Only families actually rolled out this trigger are updated; a family
        # with no observation keeps its previous estimate rather than decaying
        # toward 0, which would silently starve it.
        if _fam:
            ff = m.setdefault("_fam_fail", {})
            bf = float(CFG.FAMILY_RW_BETA)
            for _k, (_n, _bad) in _fam.items():
                if _n <= 0:
                    continue
                cur = float(_bad) / float(_n)
                ff[_k] = cur if _k not in ff else (bf * ff[_k] + (1.0 - bf) * cur)
            stats["fam_fail"] = {k: round(v, 3) for k, v in ff.items()}
        # ---- v18.5 (ii): one SGD step per observed pair ----------------------
        if CFG.CVAR_CALIBRATE:
            _cal_observe(m, [(r, y) for (r, y) in _obs if r is not None])
            _st = _cal_state(m)
            stats["cal_a"] = round(_st["a"], 4)
            stats["cal_b"] = round(_st["b"], 4)
            stats["cal_n"] = int(_st["n"])

        if minted or boosted:
            changed = True
            print("    [onpol] step %4d  minted=%d rows (buffer=%d)  yn_boosted=%d"
                  % (step, minted, len(m["onpolicy_pool"]), boosted))
    return changed


# =============================================================================
# 7c. v18.2 SPATIAL MASK: freeze most of the image
# =============================================================================
def _gaussian_blur(t, sigma):
    """Separable Gaussian on a [1,1,H,W] tensor. No torchvision dependency."""
    if sigma is None or sigma <= 0:
        return t
    rad = max(1, int(round(3.0 * float(sigma))))
    xs = torch.arange(-rad, rad + 1, device=t.device, dtype=t.dtype)
    k = torch.exp(-(xs ** 2) / (2.0 * float(sigma) ** 2))
    k = k / k.sum()
    t = F.conv2d(F.pad(t, (rad, rad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, rad, rad), mode="reflect"), k.view(1, 1, -1, 1))
    return t


@torch.no_grad()
def _mask_from_saliency(sal, keep_frac):
    """Blur -> dilate -> keep the top keep_frac of pixels. Blurring first is what
    turns scattered high-gradient pixels into coherent editable REGIONS; without
    it the mask is speckle and the perturbation still leaks across the scene."""
    sal = _gaussian_blur(sal, CFG.MASK_BLUR_SIGMA)
    d = int(CFG.MASK_DILATE or 0)
    if d and d > 1:
        if d % 2 == 0:
            d += 1
        sal = F.max_pool2d(sal, kernel_size=d, stride=1, padding=d // 2)
    n = sal.numel()
    k = int(round(float(keep_frac) * n))
    k = max(int(CFG.MASK_MIN_PIXELS), min(n, k))
    if k >= n:
        return torch.ones_like(sal)
    thr = torch.topk(sal.reshape(-1), k, largest=True).values.min()
    return (sal >= thr).to(sal.dtype)



def build_soft_entity_envelope(models, x_clean, prompt_eot, eot_samples):
    """Idea N1 (Detector-Free Soft Entity-Pair Spatial Envelope):
    Unlike v18.2's hard binary mask (which pinned 85% of pixels to 0 and starved
    SBC), this computes the clean-image gradient saliency of the relational
    pivot loss with respect to the image patches, smooths it with a wide
    Gaussian kernel (sigma = SOFT_ENV_SIGMA) to form a contiguous subject-object
    relational corridor, and returns a continuous multiplier in
    [SOFT_ENV_BG_RATIO, 1.0]:
      - Foreground entity-pair + corridor pixels (top SOFT_ENV_FG_FRAC) get 1.0 * eps
      - Un-attended background pixels get SOFT_ENV_BG_RATIO * eps (e.g. 0.35 * eps)
    This preserves unrelated background objects and reduces L_preserve / LPIPS
    while keeping 100% of the foreground attack budget on the relational pair."""
    if not getattr(CFG, "NOVEL_SOFT_ENVELOPE", False):
        return None
    if str(getattr(CFG, "BASELINE_ARM", "full")) in ("B0", "B1", "B2", "B3", "B4", "B5"):
        return None
    fg_frac = float(getattr(CFG, "SOFT_ENV_FG_FRAC", 0.70))
    bg_ratio = float(getattr(CFG, "SOFT_ENV_BG_RATIO", 0.85))
    sigma = float(getattr(CFG, "SOFT_ENV_SIGMA", 6.0))
    if fg_frac >= 1.0 or bg_ratio >= 1.0:
        return None
    x = x_clean.clone()
    _was_crn = _CRN.get("on", False)
    _CRN["on"] = False
    try:
        _, g = loss_and_grad(models, x, False, eot_samples=1, progress=0.0, need_grad=True)
    finally:
        _CRN["on"] = _was_crn
    sal = g.detach().abs().mean(dim=1, keepdim=True).float()
    if float(sal.max()) <= 0:
        return None
    sal = _gaussian_blur(sal, sigma)
    d = int(getattr(CFG, "MASK_DILATE", 9) or 0)
    if d and d > 1:
        if d % 2 == 0:
            d += 1
        sal = F.max_pool2d(sal, kernel_size=d, stride=1, padding=d // 2)
    flat = sal.reshape(-1)
    n = flat.numel()
    k_fg = max(1, min(n, int(round(fg_frac * n))))
    thr_hi = torch.topk(flat, k_fg, largest=True).values.min()
    thr_lo = flat.median()
    # Continuous ramp from bg_ratio (background) to 1.0 (entity-pair corridor)
    denom = (thr_hi - thr_lo).clamp(min=1e-8)
    ramp = ((sal - thr_lo) / denom).clamp(0.0, 1.0)
    env = bg_ratio + (1.0 - bg_ratio) * ramp
    vlog(f"    [idea-N1] soft spatial envelope: fg_frac={fg_frac:.2f} bg_ratio={bg_ratio:.2f} "
         f"mean_budget={float(env.mean()):.3f}x eps", 1)
    return env.to(x_clean.dtype)


def build_spatial_mask(models, x_clean, prompt_eot, eot_samples, keep_frac=None):
    """[1,1,H,W] mask in {0,1}: 1 = pixel may change, 0 = pinned to x_clean.

    Derived from |dL/dx| at the clean image, so it concentrates on the pixels the
    model actually uses for THIS relation between THESE two objects -- no
    bounding boxes needed, and it is model- and dataset-agnostic."""
    keep = CFG.MASK_KEEP_FRAC if keep_frac is None else keep_frac
    if (not CFG.MASK_ENABLE) or CFG.MASK_MODE == "none" or keep >= 1.0:
        return None
    x = x_clean.clone()
    _, g = loss_and_grad(models, x, prompt_eot, eot_samples=eot_samples, progress=0.0)
    sal = g.detach().abs().sum(dim=1, keepdim=True).float()
    if float(sal.max()) <= 0:
        vlog("[mask] zero gradient; mask disabled for this example", 1)
        return None
    mask = _mask_from_saliency(sal, keep)
    frac = float(mask.mean())
    vlog(f"[mask] keep_frac target={keep:.3f} actual={frac:.3f} "
         f"({int(mask.sum())} of {mask.numel()} px editable)", 1)
    return mask.to(x_clean.dtype)


# =============================================================================
# 8. APGD attack (Croce & Hein 2020), white-box.
# =============================================================================
def _apgd_checkpoints(n_iter):
    ps = [0.0, 0.22]
    while ps[-1] < 1.0:
        ps.append(ps[-1] + max(ps[-1] - ps[-2] - 0.03, 0.06))
    return [c for c in sorted({int(math.ceil(p * n_iter)) for p in ps}) if 0 < c < n_iter]


# =============================================================================
# 7b. v18.8 (3): ADAPTIVE PER-EXAMPLE BUDGET.
#
#   Coherence is a CONJUNCTION over the 7 held-out probes, so an example that
#   fails ONE probe scores 0 exactly like one that fails five -- but it is far
#   cheaper to rescue. Measured over 20 example-runs: 4 coherent, 8 failing
#   exactly one probe, 8 failing broadly. Closing only the 8 single-fail cases
#   moves coherence 0.20 -> 0.60 for +0.057 SBC. Every example currently gets
#   the same STEPS, so nothing in the optimiser can act on that distinction.
#
#   THE SIGNAL IS TRAIN-ONLY. _op_fail_frac comes from the on-policy rollout
#   over TRAIN probes. Wiring this to eval probes would be selection on the
#   test metric -- the practice the v18.2 header refuses. Do not change it.
#
#   COST NEUTRALITY IS ENFORCED, NOT ASSUMED. Extensions are paid for out of a
#   pool fed by early exits and caps. If `granted` exceeds `returned` at the end
#   of a run, this arm used more compute than its baseline and the paired delta
#   is meaningless; budget_report() says so in as many words.
# =============================================================================
_BUDGET_POOL = {"balance": 0, "granted": 0, "returned": 0,
                "decisions": [], "n_examples": 0}


def budget_reset():
    """Call once per RUN (not per example): the pool is shared across examples,
    which is the entire mechanism."""
    _BUDGET_POOL.update({"balance": 0, "granted": 0, "returned": 0,
                         "decisions": [], "n_examples": 0})


def budget_decide(models, k, n_iter, n_eff, tag=""):
    """One allocation decision. Returns (new_n_eff, stop_now, moved).

    `moved` tells apgd_attack the step schedule changed, so f_best is
    re-measured -- same contract as ONPOLICY_REFRESH_BEST and
    PROBE_REFRESH_BEST. Extending the horizon changes `progress`, which drives
    the CVaR anneal and the checkpoint grid, so the stored best refers to a
    schedule that no longer exists.
    """
    ff = models[0].get("_op_fail_frac") if models else None
    if ff is None:
        # No rollout yet. Refusing to decide is correct: deciding on a missing
        # measurement is how a "measured" allocation becomes a hidden constant.
        return n_eff, False, False
    ff = float(ff)
    n_max = int(CFG.BUDGET_MAX_MULT * n_iter)
    rec = {"step": int(k), "fail_frac": round(ff, 4), "n_eff_before": int(n_eff)}
    stop, moved = False, False

    if ff <= 0.0 and CFG.BUDGET_EARLY_EXIT:
        back = max(0, n_eff - k)
        _BUDGET_POOL["balance"] += back
        _BUDGET_POOL["returned"] += back
        rec.update({"action": "early_exit", "delta": -back})
        stop = True
    elif ff <= float(CFG.BUDGET_LOW):
        grant = min(int(CFG.BUDGET_BONUS), _BUDGET_POOL["balance"], n_max - n_eff)
        grant = max(0, grant)
        if grant > 0:
            n_eff += grant
            _BUDGET_POOL["balance"] -= grant
            _BUDGET_POOL["granted"] += grant
            moved = True
            rec.update({"action": "extend", "delta": grant})
        else:
            # Pool empty or ceiling reached. Recorded, because "the mechanism
            # wanted to act and could not" is a different outcome from "the
            # mechanism chose not to act", and only one of them is evidence
            # about the idea.
            rec.update({"action": "extend_denied", "delta": 0,
                        "pool": _BUDGET_POOL["balance"]})
    elif ff >= float(CFG.BUDGET_HIGH):
        cut = max(0, (n_eff - k) // 2)
        if cut > 0:
            n_eff -= cut
            _BUDGET_POOL["balance"] += cut
            _BUDGET_POOL["returned"] += cut
            moved = True
            rec.update({"action": "cap", "delta": -cut})
        else:
            rec.update({"action": "cap_noop", "delta": 0})
    else:
        rec.update({"action": "hold", "delta": 0})

    rec["n_eff_after"] = int(n_eff)
    rec["pool_after"] = int(_BUDGET_POOL["balance"])
    _BUDGET_POOL["decisions"].append(rec)
    if CFG.BUDGET_DEBUG:
        print(f"    [budget]{(' ' + tag) if tag else ''} step={k} "
              f"fail_frac={ff:.3f} -> {rec['action']} ({rec['delta']:+d}) "
              f"n_eff={n_eff} pool={_BUDGET_POOL['balance']}")
    return n_eff, stop, moved


def budget_report():
    """Auditable summary for results.json. `cost_neutral` is the gate: read it
    before reading any delta measured against a non-budget arm."""
    g, r = int(_BUDGET_POOL["granted"]), int(_BUDGET_POOL["returned"])
    out = {"granted": g, "returned": r, "balance": int(_BUDGET_POOL["balance"]),
           "n_decisions": len(_BUDGET_POOL["decisions"]),
           "actions": {}, "decisions": list(_BUDGET_POOL["decisions"]),
           "cost_neutral": bool(g <= r)}
    for d in _BUDGET_POOL["decisions"]:
        a = d.get("action", "?")
        out["actions"][a] = out["actions"].get(a, 0) + 1
    if not out["cost_neutral"]:
        out["WARNING"] = ("granted > returned: this arm consumed MORE optimiser "
                          "steps than its baseline. Any paired delta against a "
                          "non-budget arm is confounded with compute and must "
                          "not be reported.")
    return out


# =============================================================================
# 7d. STEP 3 BASELINES: B4 (CroPA, Luo et al. ICLR 2024) &
#                       B5 (Feature Matching, Zhao et al. NeurIPS 2023)
# =============================================================================
def _get_vision_patch_features(m, x01):
    """Extract vision-tower patch embeddings (before or after multi_modal_projector)
    differentiably with respect to x01 in [0, 1]."""
    pv = to_pixel_values(x01, m["mean"], m["std"], m["dtype"])
    model = m["model"]
    vt = getattr(model, "vision_tower", None) or getattr(getattr(model, "model", None), "vision_tower", None)
    mm_proj = getattr(model, "multi_modal_projector", None) or getattr(getattr(model, "model", None), "multi_modal_projector", None)
    out = vt(pv, output_hidden_states=True)
    feat = out.last_hidden_state
    if mm_proj is not None:
        try:
            feat = mm_proj(feat)
        except Exception:
            pass
    return feat.float()


def feature_match_attack_b5(models, x_clean, eps, n_iter, fam="lateral", tag="B5-feat"):
    """Step 3 Baseline B5 (AttackVLM-style Feature Matching, Zhao et al., NeurIPS 2023):
    Minimises MSE between the vision-encoder patch features of x_adv and the
    counterfactual image (horizontally mirrored image on lateral; spatial flip on
    vertical/other families) using APGD under the exact same L_inf budget."""
    lower = torch.clamp(x_clean - eps, 0.0, 1.0)
    upper = torch.clamp(x_clean + eps, 0.0, 1.0)
    def proj(x): return torch.max(torch.min(x, upper), lower)

    if fam == "vertical":
        x_target = x_clean.flip(-2).detach()
    else:
        x_target = x_clean.flip(-1).detach()

    with torch.no_grad():
        target_feats = [_get_vision_patch_features(m, x_target).detach() for m in models]

    crn_set_step(0, "init")
    x = proj(x_clean + (crn_rand_like(x_clean, "init", 0) - 0.5) * 2.0 * eps)
    eta = CFG.APGD_ETA_INIT_MULT * eps
    a_mom = CFG.APGD_ALPHA_MOM
    x_prev = x.clone()
    x_best = x.clone()
    f_best = float("inf")
    history = []
    t0 = time.time()
    log_every = max(1, n_iter // 10)
    checkpoints = set(_apgd_checkpoints(n_iter))
    n_improve, prev_ckpt, f_best_last, eta_last = 0, 0, float("inf"), eta
    L_prev = float("inf")

    for k in range(n_iter):
        crn_set_step(k, "main")
        xr = x.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            xt = eot_view(xr, _make_dithers(xr, 1, False)[0])
            loss = torch.zeros((), device=device)
            for m, tf in zip(models, target_feats):
                cur_f = _get_vision_patch_features(m, xt)
                loss = loss + F.mse_loss(cur_f, tf)
            g, = torch.autograd.grad(loss, xr)
        L_val = float(loss.detach())
        history.append(L_val)
        if L_val < f_best:
            f_best, x_best = L_val, x.detach().clone()
        if k > 0 and L_val < L_prev:
            n_improve += 1
        L_prev = L_val
        z = proj(x - eta * g.detach().sign())
        x_next = proj(x + a_mom * (z - x) + (1.0 - a_mom) * (x - x_prev))
        x_prev, x = x, x_next
        if k in checkpoints:
            interval = max(1, k - prev_ckpt)
            if (n_improve < CFG.APGD_RHO * interval) or (eta == eta_last and f_best == f_best_last):
                eta /= 2.0
                x = x_best.clone(); x_prev = x_best.clone(); L_prev = f_best
            n_improve, prev_ckpt, f_best_last, eta_last = 0, k, f_best, eta
        if k % log_every == 0 or k == n_iter - 1:
            print(f"    [{tag}] step {k:4d}/{n_iter}  feat_mse={L_val:.5f}  best={f_best:.5f}  eta={eta:.4f}")
    result = quantize_project(x_best, x_clean, eps).detach()
    crn_archive()
    print(f"[attack] {tag} done in {time.time()-t0:.1f}s | best_mse={f_best:.5f}")
    return result, history


def cropa_attack_b4(models, x_clean, eps, n_iter, prompt_eot, eot_samples=None, tag="B4-CroPA"):
    """Step 3 Baseline B4 (CroPA, Luo et al., ICLR 2024):
    Cross-Prompt Attack via adversarial prompt perturbation in the language
    embedding space. Jointly updates an adversarial perturbation delta_p on the
    text token embeddings (maximising/diversifying prompt variation) while
    minimising the targeted cross-entropy loss over the image perturbation x_adv."""
    m0 = models[0]
    emb_layer = m0["model"].get_input_embeddings()
    orig_fwd = emb_layer.forward
    state_p = {"delta": None}

    def _hooked_emb_fwd(input_ids):
        out = orig_fwd(input_ids)
        if state_p["delta"] is not None and out.shape[1] > 16:
            plen = min(state_p["delta"].shape[1], out.shape[1] - 1)
            d = state_p["delta"][:, -plen:, :].to(dtype=out.dtype, device=out.device)
            out = out.clone()
            out[:, -plen:, :] = out[:, -plen:, :] + d
        return out

    emb_layer.forward = _hooked_emb_fwd
    try:
        hidden_dim = int(emb_layer.weight.shape[1])
        plen = int(getattr(CFG, "CROPA_PROMPT_LEN", 8))
        p_lr = float(getattr(CFG, "CROPA_PROMPT_LR", 0.05))
        p_eps = 0.10
        delta_p = torch.zeros((1, plen, hidden_dim), device=device, dtype=torch.float32)

        lower = torch.clamp(x_clean - eps, 0.0, 1.0)
        upper = torch.clamp(x_clean + eps, 0.0, 1.0)
        def proj(x): return torch.max(torch.min(x, upper), lower)

        crn_set_step(0, "init")
        x = proj(x_clean + (crn_rand_like(x_clean, "init", 0) - 0.5) * 2.0 * eps)
        eta = CFG.APGD_ETA_INIT_MULT * eps
        a_mom = CFG.APGD_ALPHA_MOM
        x_prev = x.clone()
        x_best, f_best = x.clone(), float("inf")
        history = []
        t0 = time.time()
        log_every = max(1, n_iter // 10)
        checkpoints = set(_apgd_checkpoints(n_iter))
        n_improve, prev_ckpt, f_best_last, eta_last = 0, 0, float("inf"), eta
        L_prev = float("inf")

        for k in range(n_iter):
            crn_set_step(k, "main")
            dp = delta_p.detach().clone().requires_grad_(True)
            state_p["delta"] = dp
            xr = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                xt = eot_view(xr, _make_dithers(xr, 1, False)[0])
                pv = to_pixel_values(xt, m0["mean"], m0["std"], m0["dtype"])
                probes = _active_probes(m0, prompt_eot)
                wmap = {p["pkey"]: CFG.LAMBDA_CONSIST for p in probes}
                Lc, _ = _loss_batched(m0, pv, probes, wmap)
                gx, gp = torch.autograd.grad(Lc, [xr, dp])
            delta_p = (delta_p + p_lr * gp.detach().sign()).clamp(-p_eps, p_eps)
            state_p["delta"] = None

            L_val = float(Lc.detach())
            history.append(L_val)
            if L_val < f_best:
                f_best, x_best = L_val, x.detach().clone()
            if k > 0 and L_val < L_prev:
                n_improve += 1
            L_prev = L_val
            z = proj(x - eta * gx.detach().sign())
            x_next = proj(x + a_mom * (z - x) + (1.0 - a_mom) * (x - x_prev))
            x_prev, x = x, x_next
            if k in checkpoints:
                interval = max(1, k - prev_ckpt)
                if (n_improve < CFG.APGD_RHO * interval) or (eta == eta_last and f_best == f_best_last):
                    eta /= 2.0
                    x = x_best.clone(); x_prev = x_best.clone(); L_prev = f_best
                n_improve, prev_ckpt, f_best_last, eta_last = 0, k, f_best, eta
            if k % log_every == 0 or k == n_iter - 1:
                print(f"    [{tag}] step {k:4d}/{n_iter}  loss={L_val:.4f}  best={f_best:.4f}  eta={eta:.4f}")

        result = quantize_project(x_best, x_clean, eps).detach()
        crn_archive()
        print(f"[attack] {tag} done in {time.time()-t0:.1f}s | loss-best={f_best:.4f}")
        return result, history
    finally:
        emb_layer.forward = orig_fwd


def apgd_attack(models, x_clean, eps, n_iter, prompt_eot, random_init=True,
                tag="cw-ste", eot_samples=None, mask=None):
    if mask is not None and bool(((mask > 0.0) & (mask < 1.0)).any()):
        # Idea N1: Soft spatial envelope -> per-pixel L_inf radius eps(u,v) = eps * mask(u,v)
        eps_map = eps * mask
        lower = torch.clamp(x_clean - eps_map, 0.0, 1.0)
        upper = torch.clamp(x_clean + eps_map, 0.0, 1.0)
        def proj(x):
            return torch.max(torch.min(x, upper), lower)
    else:
        lower = torch.clamp(x_clean - eps, 0.0, 1.0)
        upper = torch.clamp(x_clean + eps, 0.0, 1.0)
        def proj(x):
            x = torch.max(torch.min(x, upper), lower)
            if mask is not None:
                x = x_clean + mask * (x - x_clean)
            return x

    def loss_grad(x, progress=0.0, phase=None, need_grad=True):
        """`phase` tags the CRN substream. Refresh calls (mint / dual) use
        `need_grad=False` (forward-only under torch.no_grad()), eliminating the
        unused 13B backward pass on x_best while keeping f_best exact."""
        if phase is not None:
            prev = _CRN.get("phase", "main")
            crn_set_step(_CRN.get("step", 0), phase)
            try:
                return loss_and_grad(models, x, prompt_eot,
                                     eot_samples=eot_samples, progress=progress,
                                     need_grad=need_grad)
            finally:
                crn_set_step(_CRN.get("step", 0), prev)
        return loss_and_grad(models, x, prompt_eot, eot_samples=eot_samples,
                             progress=progress, need_grad=need_grad)

    # v18.7 IDEA-B: the random init is drawn from the pinned stream, so two
    # arms start from the BIT-IDENTICAL point. Under the old code the init came
    # off the global RNG whose position already depended on how many probe
    # subsets had been sampled during setup.
    crn_set_step(0, "init")
    if random_init:
        if CFG.CRN_ENABLE:
            x = proj(x_clean + (crn_rand_like(x_clean, "init", 0) - 0.5) * 2.0 * eps)
            crn_fingerprint("x_init", x - x_clean)
        else:
            # EXACT v18.6 call. `uniform_` and `(rand_like - 0.5)*2*eps` draw
            # from the same distribution but are DIFFERENT kernels consuming the
            # RNG differently, so they return different numbers from the same
            # seed. Keeping the original call here is what makes CRN_ENABLE=False
            # bit-identical to v18.6 -- otherwise the `crn_off` control arm would
            # itself be a new draw, and the whole point of that arm is to
            # reproduce the pre-CRN noise, not to add to it.
            x = proj(x_clean + torch.empty_like(x_clean).uniform_(-eps, eps))
    else:
        x = x_clean.clone()

    # v18.7 IDEA-A: fresh duals for this run, and a visible statement of the
    # constraint being solved.
    if CFG.PROBE_LAGRANGIAN:
        for _m in models:
            init_probe_lagrangian(_m)
        print(f"    [lagr] per-probe Lagrangian ON: min||d|| s.t. L_p <= "
              f"{CFG.PROBE_C} for all p  |  lr={CFG.PROBE_LAMBDA_LR} "
              f"lam0={CFG.PROBE_LAMBDA_INIT} lam_max={CFG.PROBE_LAMBDA_MAX} "
              f"normalize={CFG.PROBE_LAMBDA_NORMALIZE}")
        print(f"    [lagr] CVaR/hardprobe/FMR are BYPASSED while this is on "
              f"(dro_cvar={CFG.DRO_CVAR} hardprobe={CFG.HARDPROBE_WEIGHTING} "
              f"famrw={CFG.FAMILY_REWEIGHT} -> all inert)")

    crn_set_step(0, "main")
    L, g = loss_grad(x, 0.0)
    checkpoints = set(_apgd_checkpoints(n_iter))
    eta = CFG.APGD_ETA_INIT_MULT * eps; a_mom = CFG.APGD_ALPHA_MOM
    x_best, f_best = x.clone(), L.clone()
    x_prev = x.clone(); x = proj(x - eta * g.sign())
    n_improve, f_best_last, eta_last, prev_ckpt = 0, f_best.clone(), eta, 0
    # v18.3 (D): Croce & Hein's condition 1 counts improvement over the PREVIOUS
    # ITERATE, sum_i 1[f(x^(i+1)) > f(x^(i))].  v18.2 counted improvement over
    # the running BEST, which is strictly harder to satisfy, so cond1 fired too
    # often and eta halved prematurely.  n_scored tracks how many iterations in
    # the current interval were actually comparable, because iterations whose
    # objective just changed are excluded rather than judged against a stale
    # loss (the objective moves here: minting, and the preservation dual).
    L_prev = L.clone()
    n_scored = 0
    cond1_mode = getattr(CFG, "APGD_COND1_MODE", "prev")

    history = [float(L)]; log_every = max(1, n_iter // 10); t0 = time.time()
    lam_history = []
    empty_every = max(0, getattr(CFG, "EMPTY_CACHE_EVERY", 0))
    # op_start and the n_iter passed to _onpolicy_update stay pinned to the
    # ORIGINAL horizon, so the minting schedule up to the budget decision is
    # identical to the baseline arm. Only the loop bound, `progress` and the
    # checkpoint grid follow n_eff.
    op_start = int(CFG.ONPOLICY_START_FRAC * n_iter)
    op_every = max(1, int(CFG.ONPOLICY_EVERY))
    # v18.8 (3): the horizon is now mutable. `n_eff` starts equal to n_iter, so
    # with BUDGET_ADAPT=False this loop is exactly the v18.7 for-loop.
    n_eff = int(n_iter)
    budget_k = (int(CFG.BUDGET_DECIDE_FRAC * n_iter) if CFG.BUDGET_ADAPT else -1)
    budget_done = False
    steps_used = 0
    k = 0
    while True:
        k += 1
        if k >= n_eff:
            break
        steps_used = k
        progress = k / float(max(1, n_eff - 1))
        objective_changed = False
        # v18.7 IDEA-B: every stochastic draw below is now a function of k, not
        # of stream position. Extending the horizon therefore does NOT
        # desynchronise this arm from its comparison arm: the dither at step k
        # is the same tensor either way. Without IDEA-B this idea would be
        # unmeasurable.
        crn_set_step(k, "main")

        # ---- IDEA-2: roll out and mint rows BEFORE this step's gradient.
        if CFG.ONPOLICY and k >= op_start and k % op_every == 0:
            if _onpolicy_update(models, x, k, n_iter):
                objective_changed = True
                if CFG.ONPOLICY_REFRESH_BEST:
                    Lb, _ = loss_grad(x_best, progress, phase="refresh_mint", need_grad=False)
                    f_best = Lb.detach().clone()
                    f_best_last = f_best.clone()

        # ---- v18.8 (3): the allocation decision.
        if CFG.BUDGET_ADAPT and (not budget_done) and k >= budget_k:
            n_new, _stop, _moved = budget_decide(models, k, n_iter, n_eff, tag=tag)
            if _moved or _stop:
                budget_done = True
            if n_new != n_eff:
                n_eff = n_new
                checkpoints = set(_apgd_checkpoints(n_eff))
                log_every = max(1, n_eff // 10)
                objective_changed = True
                Lb, _ = loss_grad(x_best, progress, phase="refresh_budget", need_grad=False)
                f_best = Lb.detach().clone()
                f_best_last = f_best.clone()
            if _stop:
                break

        L, g = loss_grad(x, progress, need_grad=True); history.append(float(L))

        # ---- v18.7 IDEA-A & v18.3 (A): dual ascent on per-probe and preserve
        # constraints. Coalesce their f_best refresh so at most ONE forward-only
        # refresh runs after the gradient step.
        _post_refresh_phase = None
        if CFG.PROBE_LAGRANGIAN and (k % max(1, int(CFG.PROBE_LAMBDA_EVERY)) == 0):
            if update_probe_lambdas(models[0], step=k):
                objective_changed = True
                if CFG.PROBE_REFRESH_BEST:
                    models[0]["_probe_lag_events"]["refresh"] += 1
                    _post_refresh_phase = "refresh_lagr"
            if CFG.PROBE_LAG_DEBUG and (k % max(1, int(CFG.PROBE_LAG_DEBUG_EVERY)) == 0):
                _dbg_probe_lag(models[0], k, n_eff)

        if CFG.PRESERVE_LOSS and CFG.PRESERVE_LAGRANGIAN:
            if update_preserve_lambda(models):
                objective_changed = True
                if CFG.PRESERVE_REFRESH_BEST and _post_refresh_phase is None:
                    _post_refresh_phase = "refresh_pres"
            lam_history.append(float(models[0].get("preserve_lambda", 0.0)))

        if _post_refresh_phase is not None:
            Lb, _ = loss_grad(x_best, progress, phase=_post_refresh_phase, need_grad=False)
            f_best = Lb.detach().clone()
            f_best_last = f_best.clone()

        # ---- improvement bookkeeping (global best is still tracked, because
        # x_best is what we return; only the cond1 COUNTER changes).
        improved_prev = bool(L < L_prev)
        if L < f_best:
            f_best, x_best = L.clone(), x.clone()
        skip = objective_changed and getattr(CFG, "APGD_SKIP_CHANGED_STEPS", True)
        if not skip:
            n_scored += 1
            if cond1_mode == "prev":
                n_improve += int(improved_prev)
            else:                                   # "best" == v18.2 behaviour
                n_improve += int(bool(L < f_best_last))
        L_prev = L.clone()

        z = proj(x - eta * g.sign())
        x_next = proj(x + a_mom * (z - x) + (1.0 - a_mom) * (x - x_prev))
        x_prev, x = x, x_next
        if k in checkpoints:
            # Denominator is the number of COMPARABLE iterations, not the raw
            # interval; otherwise excluding a mint step would itself look like a
            # failure to improve and halve eta for the wrong reason.
            interval = n_scored if getattr(CFG, "APGD_SKIP_CHANGED_STEPS", True) \
                else (k - prev_ckpt)
            interval = max(1, interval)
            cond1 = n_improve < CFG.APGD_RHO * interval
            cond2 = (eta == eta_last) and (float(f_best) == float(f_best_last))
            if cond1 or cond2:
                eta = eta / 2.0; x = x_best.clone(); x_prev = x_best.clone()
                # We just teleported back to x_best, so the next iterate must be
                # compared against f(x_best), not against the loss at the point
                # we abandoned -- otherwise the restart itself registers as a
                # large "improvement" and inflates n_improve for free.
                L_prev = f_best.clone()
            n_improve = 0; n_scored = 0
            f_best_last = f_best.clone(); eta_last = eta; prev_ckpt = k
        if empty_every and (k % empty_every == 0) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if k % log_every == 0 or k == n_eff - 1:
            _na = len(models[0].get("onpolicy_pool", [])) if models else 0
            _mf = "" if mask is None else f"  mask={float(mask.mean()):.2f}"
            _pv = models[0].get("_preserve_last") if models else None
            _pl = ("  pres=%.3f lam=%.2f" % (_pv, models[0].get("preserve_lambda", 0.0))
                   if (_pv is not None and CFG.PRESERVE_LOSS) else "")
            # v18.5 telemetry: the measured failure fraction driving alpha, and
            # the calibrator's fitted slope. If fhat never moves, ONPOLICY is
            # off or has not triggered yet and CVAR_ADAPTIVE is inert.
            _cv = ""
            if models and CFG.DRO_CVAR:
                _ff = models[0].get("_op_fail_frac")
                if CFG.CVAR_ADAPTIVE and _ff is not None:
                    _cv += " fhat=%.2f" % float(_ff)
                if CFG.CVAR_CALIBRATE:
                    _cs = models[0].get("_cal") or {}
                    _cv += " cal(a=%.2f,n=%d)" % (float(_cs.get("a", 0.0)),
                                                  int(_cs.get("n", 0)))
            if models and CFG.FAMILY_REWEIGHT:
                _fm = models[0].get("_fam_mult_last") or {}
                if _fm:
                    _cv += " fam(" + ",".join("%s=%.2f" % (k[:1], v)
                                              for k, v in sorted(_fm.items())) + ")"
            print(f"    [{tag}]{_mf} step {k:4d}/{n_eff}  loss={float(L):.4f}  eta={eta:.4f}  "
                  f"Linf={(x-x_clean).abs().max().item():.4f}  best={float(f_best):.4f}  "
                  f"cvar_a={_cvar_alpha(progress, models[0] if models else None):.2f}"
                  f"{_cv}  op_rows={_na}{_pl}")

    result = quantize_project(x_best, x_clean, eps).detach()
    if mask is not None:
        if bool(((mask > 0.0) & (mask < 1.0)).any()):
            result = torch.max(torch.min(result, upper), lower).detach()
        else:
            result = (x_clean + mask * (result - x_clean)).detach()
    _pl = ""
    if CFG.PRESERVE_LOSS and models and models[0].get("preserve_pool"):
        _pl = (" | preserve=%.3f (C=%.2f) lambda=%.3f"
               % (models[0].get("_preserve_last") or float("nan"),
                  CFG.PRESERVE_C, models[0].get("preserve_lambda", 0.0)))
    vlog(f"[attack] {tag} done in {time.time()-t0:.1f}s | loss-best={float(f_best):.4f}{_pl}", 1)
    # ---- v18.7 IDEA-A: the certificate table is the deliverable, not a log.
    if CFG.PROBE_LAGRANGIAN and models:
        _print_probe_certificates(models[0], tag=f"[{tag}]")
    # ---- v18.7 IDEA-B: archive BEFORE returning. RUN_RANDOM_CONTROL calls
    # set_example_seed() again after this, which would otherwise wipe the
    # counters before the results record reads them.
    crn_archive()
    if CFG.CRN_ENABLE and CFG.CRN_DEBUG:
        print(f"    [crn] run complete: {_CRN['draws']} pinned draws, "
              f"{len(_CRN['fingerprints'])} fingerprints recorded. "
              f"Paired arms must match BOTH.")
    # history stays a plain list of floats -- craft_phase reads history[-1] for
    # final_loss. The lambda trace rides along on the model dict so nothing
    # downstream has to change shape.
    if models:
        models[0]["_lambda_history"] = lam_history
    return result, history

def random_perturbation(x_clean, eps):
    x = torch.clamp(x_clean + torch.empty_like(x_clean).uniform_(-eps, eps), 0.0, 1.0)
    return quantize_project(x, x_clean, eps)


# =============================================================================
# 9. Evaluation (SBC + by-kind SBC + strict coherence + preservation)
# =============================================================================
_LEGACY_7_PKEYS = {"o19", "o13", "o4", "y6", "n6", "y11", "n11"}


def _sbc_and_coherence(primary, image_pil, adv01, eval_probes, rt, rg,
                       collect_rows=False, subj=None, obj=None):
    pv = to_pixel_values(adv01, primary["mean"], primary["std"], primary["dtype"])
    per_kind = {"open": [0, 0], "yes": [0, 0], "no": [0, 0]}
    all_ok, hits, rows = True, 0, []
    n_degen, hits_nd, all_ok_nd = 0, 0, True
    hits_7, n_7, all_ok_7 = 0, 0, True
    per_frame = {}
    open_idx = [i for i, _t in enumerate(eval_probes) if _t[1] == "open"]
    yn_idx   = [i for i, _t in enumerate(eval_probes) if _t[1] != "open"]
    answers = [None] * len(eval_probes)
    if open_idx:
        for i, a in zip(open_idx, generate_batch(
                primary, image_pil, [eval_probes[i][0] for i in open_idx],
                pv, max_new_tokens=CFG.GEN_MAX_NEW_TOKENS)):
            answers[i] = a
    if yn_idx:
        for i, a in zip(yn_idx, generate_batch(
                primary, image_pil, [eval_probes[i][0] for i in yn_idx],
                pv, max_new_tokens=12)):
            answers[i] = a
    for _t, a in zip(eval_probes, answers):
        q, kind, expected = _t[0], _t[1], _t[2]
        pkey = _t[3] if len(_t) > 3 else None
        ok, why = score_lie(a, kind, rt, rg, subj=subj, obj=obj, with_reason=True)
        degen = (why == "degenerate") or is_degenerate(a, subj, obj)
        ok_nd = bool(ok and not degen)
        per_kind[kind][0] += int(ok); per_kind[kind][1] += 1
        hits += int(ok); all_ok = all_ok and ok
        hits_nd += int(ok_nd); all_ok_nd = all_ok_nd and ok_nd
        n_degen += int(degen)
        if pkey in _LEGACY_7_PKEYS:
            hits_7 += int(ok); n_7 += 1; all_ok_7 = all_ok_7 and ok
        if CFG.EVAL_PER_FRAME and pkey is not None:
            e = per_frame.setdefault(pkey, {"kind": kind, "hit": 0, "n": 0})
            e["hit"] += int(ok); e["n"] += 1
        if collect_rows:
            row = {"question": q, "kind": kind, "adv": a,
                   "probe_key": pkey,
                   "consistent_with_lie": ok, "degenerate": degen,
                   "consistent_nondegen": ok_nd}
            if CFG.SCORER_AUDIT:
                row["scorer_reason"] = why
            rows.append(row)
    n = max(1, len(eval_probes))
    ks = {k: (v[0] / v[1] if v[1] else None) for k, v in per_kind.items()}
    res = {"sbc": hits / n, "coherence": 1.0 if all_ok else 0.0,
           "sbc_open": ks["open"], "sbc_yes": ks["yes"], "sbc_no": ks["no"],
           "sbc_nondegen": hits_nd / n,
           "coherence_nondegen": 1.0 if all_ok_nd else 0.0,
           "n_degenerate": n_degen,
           "sbc_7": (hits_7 / n_7) if n_7 > 0 else (hits / n),
           "coherence_7": (1.0 if all_ok_7 else 0.0) if n_7 > 0 else (1.0 if all_ok else 0.0)}
    if CFG.EVAL_PER_FRAME and per_frame:
        res["per_frame"] = {k: {"kind": v["kind"], "hit": v["hit"], "n": v["n"],
                                "acc": round(v["hit"] / max(1, v["n"]), 4)}
                            for k, v in sorted(per_frame.items())}
    if collect_rows: res["rows"] = rows
    return res


_LPIPS_METRIC = [None]


@torch.no_grad()
def compute_perceptual_dist(clean01, adv01):
    """Step 3 Main Table metric: LPIPS perceptual distance (or normalised L2 fallback
    if the `lpips` package is not installed)."""
    if _LPIPS_METRIC[0] is None:
        try:
            import lpips  # type: ignore
            _LPIPS_METRIC[0] = lpips.LPIPS(net="alex").to(device).eval()
        except Exception:
            _LPIPS_METRIC[0] = False
    if _LPIPS_METRIC[0]:
        c_m11 = clean01.float() * 2.0 - 1.0
        a_m11 = adv01.float() * 2.0 - 1.0
        return round(float(_LPIPS_METRIC[0](c_m11, a_m11).mean().item()), 5)
    diff = (clean01.float() - adv01.float())
    return round(float(diff.pow(2).mean().sqrt().item()), 5)


def evaluate_full(primary, image_pil, clean01, adv01, eval_probes, preserve_qs, rt, rg,
                  subj=None, obj=None):
    ev = _sbc_and_coherence(primary, image_pil, adv01, eval_probes, rt, rg,
                            collect_rows=True, subj=subj, obj=obj)
    pv_clean = to_pixel_values(clean01, primary["mean"], primary["std"], primary["dtype"])
    pv_adv   = to_pixel_values(adv01,   primary["mean"], primary["std"], primary["dtype"])
    strict, f1_sum = 0, 0.0
    # Reuse clean preservation answers already frozen in build_preserve_rows when available
    cached_pres = {r["q"]: r["answer"] for r in (primary.get("preserve_pool") or [])}
    missing_clean_qs = [q for q in preserve_qs if q not in cached_pres]
    if missing_clean_qs:
        for q, a_c in zip(missing_clean_qs, generate_batch(primary, image_pil, missing_clean_qs, pv_clean)):
            cached_pres[q] = a_c
    adv_pres_ans = generate_batch(primary, image_pil, list(preserve_qs), pv_adv)
    for q, a_adv in zip(preserve_qs, adv_pres_ans):
        a_clean = cached_pres[q]
        f1 = token_f1(a_clean, a_adv)
        strict += int(a_clean.strip().lower() == a_adv.strip().lower()); f1_sum += f1
        ev["rows"].append({"type": "preserve", "question": q, "clean": a_clean,
                           "adv": a_adv, "token_f1": round(f1, 3)})
    n = max(1, len(preserve_qs))
    ev["preserve_strict"] = strict / n; ev["preserve_f1"] = f1_sum / n

    # ---- v18.8 (2): THE CLEAN-IMAGE CONTROL -------------------------------
    # sbc_yes sat at 0.975 across every run. That is NOT evidence the attack
    # works if the clean model assents to the same confirmation frames anyway:
    # a "yes" probe asks whether the TARGET (false) relation holds, and a model
    # with an assent prior answers "Yes." to any confirmation regardless of the
    # image. Score the identical held-out probes on the CLEAN image and report
    # the LIFT. SBC_lift, not SBC, is the quantity attributable to the attack.
    if CFG.EVAL_CLEAN_CONTROL:
        cc = _sbc_and_coherence(primary, image_pil, clean01, eval_probes, rt, rg,
                                collect_rows=True, subj=subj, obj=obj)
        ev["SBC_clean"]       = cc["sbc"]
        ev["coherence_clean"] = cc["coherence"]
        ev["sbc_open_clean"]  = cc["sbc_open"]
        ev["sbc_yes_clean"]   = cc["sbc_yes"]
        ev["sbc_no_clean"]    = cc["sbc_no"]
        ev["SBC_lift"]        = ev["sbc"] - cc["sbc"]
        ev["sbc_yes_lift"]    = ((ev["sbc_yes"] - cc["sbc_yes"])
                                 if (ev["sbc_yes"] is not None
                                     and cc["sbc_yes"] is not None) else None)
        ev["sbc_no_lift"]     = ((ev["sbc_no"] - cc["sbc_no"])
                                 if (ev["sbc_no"] is not None
                                     and cc["sbc_no"] is not None) else None)
        # PGCM Stage-0 diagnostic: open-question lift = adv - clean. This is the
        # quantity the mirror teacher is meant to raise (and must never lower).
        # Read it per example alongside cf_info.{open_dir_on, cf_beta, cf_m0} to
        # see directly whether the teacher fired and whether open accuracy held.
        ev["sbc_open_lift"]   = ((ev["sbc_open"] - cc["sbc_open"])
                                 if (ev.get("sbc_open") is not None
                                     and cc.get("sbc_open") is not None) else None)
        if "per_frame" in cc:
            ev["per_frame_clean"] = cc["per_frame"]
        ev["clean_probe_log"] = cc.get("rows", [])
    return ev

# =============================================================================
# LATERAL OPEN-ANSWER DIAGNOSTICS (used by OPEN_DIAG and mirror_probe.py)
#   Looks BELOW the greedy answer, at two slots of a teacher-forced answer:
#   * DODGE slot     "The <s> is ___"         -> mass on next/near/beside/...
#   * DIRECTION slot "The <s> is to the ___"  -> log p(target dir) - log p(true dir)
# =============================================================================
DODGE_WORDS = ["next", "near", "beside", "close", "located"]


def _last_tok(tok, text):
    ids = tok(text, add_special_tokens=False).input_ids
    return ids[-1] if ids else None


@torch.no_grad()
def _slot_logp(m, image_pil, pv, question, answer):
    """log-probs (over the vocab) of the LAST answer token's slot, i.e. the
    model's distribution for what follows answer[:-1]."""
    ids, attn, alen = encode_probe(m["processor"], image_pil, question, answer)
    out = m["model"](input_ids=ids, attention_mask=attn, pixel_values=pv, use_cache=False)
    return F.log_softmax(out.logits[0, -2, :].float(), dim=-1)


def _open_dir_slot_spec(tok, subj, rt, rg):
    """Return (carrier_prefix, probe_str, i_true, i_tgt) for the open-answer
    direction slot of any spatial relation family."""
    if relation_family(rt) == "lateral":
        d_true, d_tgt = relation_group(rt), relation_group(rg)
        carrier = "The %s is to the" % subj
        probe_str = "The %s is to the %s" % (subj, d_true)
        i_true = _last_tok(tok, "The %s is to the %s" % (subj, d_true))
        i_tgt = _last_tok(tok, "The %s is to the %s" % (subj, d_tgt))
        return carrier, probe_str, i_true, i_tgt
    carrier = "The %s is" % subj
    _canon = {"ahead of": "in front of", "beneath": "below", "on top of": "above"}
    rt_s = _canon.get(str(rt).strip().lower(), str(rt).strip())
    rg_s = _canon.get(str(rg).strip().lower(), str(rg).strip())
    ids_c = tok(carrier, add_special_tokens=False).input_ids
    ids_rt = tok(carrier + " " + rt_s, add_special_tokens=False).input_ids
    ids_rg = tok(carrier + " " + rg_s, add_special_tokens=False).input_ids
    i_true = ids_rt[len(ids_c)] if len(ids_rt) > len(ids_c) else _last_tok(tok, str(rt))
    i_tgt = ids_rg[len(ids_c)] if len(ids_rg) > len(ids_c) else _last_tok(tok, str(rg))
    return carrier, "The %s is next" % subj, i_true, i_tgt


@torch.no_grad()
def lateral_slot_stats(m, image_pil, pv, question, subj, rt, rg, topk=3):
    """Dodge mass, top tokens, and direction gap for one open question across
    all 5 spatial relation families (bit-identical to v18.11 on lateral)."""
    tok = m["processor"].tokenizer
    lp = _slot_logp(m, image_pil, pv, question, "The %s is next" % subj)
    dodge_ids = {_last_tok(tok, "The %s is %s" % (subj, w)) for w in DODGE_WORDS}
    dodge_ids.discard(None)
    dodge_mass = float(lp[list(dodge_ids)].exp().sum())
    top = torch.topk(lp, topk)
    top_tokens = [(tok.decode([int(i)]).strip(), round(float(v.exp()), 3))
                  for v, i in zip(top.values, top.indices)]
    _, probe_str, i_true, i_tgt = _open_dir_slot_spec(tok, subj, rt, rg)
    lp2 = lp if probe_str == ("The %s is next" % subj) else _slot_logp(
        m, image_pil, pv, question, probe_str)
    dir_gap = float(lp2[i_tgt] - lp2[i_true])
    dir_mass = float(lp2[i_tgt].exp() + lp2[i_true].exp())
    return {"dodge_mass": round(dodge_mass, 4), "top": top_tokens,
            "dir_gap": round(dir_gap, 3), "dir_mass": round(dir_mass, 4),
            "tok_true": tok.decode([i_true]).strip(), "tok_target": tok.decode([i_tgt]).strip()}


axis_slot_stats = lateral_slot_stats


@torch.no_grad()
def yesno_gap(m, image_pil, pv, question):
    """log p(Yes) - log p(No) for the first answer token."""
    tok = m["processor"].tokenizer
    lp = _slot_logp(m, image_pil, pv, question, "Yes")
    i_yes, i_no = _last_tok(tok, "Yes"), _last_tok(tok, "No")
    return round(float(lp[i_yes] - lp[i_no]), 3)


@torch.no_grad()
def yesno_gap_batch(m, image_pil, pv, questions):
    """Batched log p(Yes) - log p(No) across all `questions` in ONE forward pass
    (or chunks of MAX_ROWS_PER_FWD). Replaces 24-112 sequential batch=1 forwards
    in _mbpd_table and mbpd_diag with 1-2 batched forwards."""
    if not questions:
        return []
    tok = m["processor"].tokenizer
    i_yes, i_no = _last_tok(tok, "Yes"), _last_tok(tok, "No")
    encoded = [encode_probe(m["processor"], image_pil, q, "Yes") for q in questions]
    chunk_sz = max(8, int(getattr(CFG, "MAX_ROWS_PER_FWD", 12)))
    out_gaps = []
    for s in range(0, len(encoded), chunk_sz):
        sub = encoded[s:s + chunk_sz]
        B = len(sub)
        Lmax = max(r[0].shape[1] for r in sub)
        padded = torch.full((B, Lmax), m["pad_id"], dtype=torch.long, device=device)
        attn   = torch.zeros((B, Lmax), dtype=torch.long, device=device)
        for j, (ids, _a, _al) in enumerate(sub):
            L = ids.shape[1]
            padded[j, -L:] = ids[0]
            attn[j, -L:] = 1
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        out = _model_fwd_tail(m["model"], input_ids=padded, attention_mask=attn,
                              position_ids=pos,
                              pixel_values=pv.expand(B, -1, -1, -1),
                              logits_to_keep=2)
        lp = F.log_softmax(out.logits[:, -2, :].float(), dim=-1)
        gaps = (lp[:, i_yes] - lp[:, i_no]).tolist()
        out_gaps.extend(round(float(g), 3) for g in gaps)
        del out, lp
    return out_gaps


def _mbpd_classify(gc, gm, sgn, marg, pmin, amax):
    """Shared by the v18.10 diagnostic and the v18.11 loss. Returns
    (status, alpha, P_signed). sgn=+1 target-Q (lie = Yes), -1 true-Q (lie = No)."""
    Ps = sgn * 0.5 * (gm - gc)
    if sgn * gc >= marg:
        return "prior_already_lie", 0.0, Ps
    if sgn * gm >= marg:
        return "mirror_enough", 0.0, Ps
    if Ps < pmin:
        return "perception_dead", None, Ps
    alpha = (marg - sgn * gc) / (2.0 * Ps) - 1.0
    return ("beyond_mirror" if alpha <= amax else "beyond_mirror_capped"), alpha, Ps


@torch.no_grad()
def _mbpd_table(m, image_pil, pv_c, pv_m, subj, obj, rt, rg, cf_op=None):
    """v18.11 + DICD + Batched Yes/No Gaps + Symmetric Shortfall Refinements."""
    if cf_op is None:
        fam = relation_family(rt)
        cf_mode_cfg = str(getattr(CFG, "CF_COUNTERFACTUAL_MODE", "auto")).lower()
        cf_op = "mirror" if (fam == "lateral" and cf_mode_cfg != "dual") else "dual"
    marg = float(getattr(CFG, "MBPD_MARGIN", 1.0))
    pmin = float(getattr(CFG, "MBPD_P_MIN", 0.5))
    amax = float(getattr(CFG, "MBPD_ALPHA_MAX", 8.0))
    cap = float(getattr(CFG, "MBPD_ALPHA_CAP", 3.0))
    fw = float(getattr(CFG, "MBPD_FRAG_W", 0.65))
    p0 = float(getattr(CFG, "MBPD_FRAG_P0", 0.75))
    ps = max(1e-3, float(getattr(CFG, "MBPD_FRAG_S", 0.25)))
    use_shortfall = bool(getattr(CFG, "MBPD_SHORTFALL_WEIGHT", True))
    sym_frag = bool(getattr(CFG, "MBPD_SYMMETRIC_FRAG", True))
    dead_assent_teach = bool(getattr(CFG, "MBPD_DEAD_ASSENT_TEACHER", True))
    dead_p_floor = float(getattr(CFG, "MBPD_DEAD_P_FLOOR", 0.05))
    train_fis = list(POOL["yn_train"])
    qs = []
    for fi in train_fis:
        frame = YESNO_FRAMES[fi]
        qs.append(frame.format(s=subj, o=obj, r=rg))
        qs.append(frame.format(s=subj, o=obj, r=rt))
    gaps_c = yesno_gap_batch(m, image_pil, pv_c, qs)
    gaps_m = (yesno_gap_batch(m, image_pil, pv_m, qs)
              if cf_op == "mirror" else None)
    tab = {}
    for idx, fi in enumerate(train_fis):
        gc_yes = float(gaps_c[2 * idx])
        gc_no  = float(gaps_c[2 * idx + 1])
        if cf_op == "mirror":
            gm_yes = float(gaps_m[2 * idx])
            gm_no  = float(gaps_m[2 * idx + 1])
        else:
            gm_yes, gm_no = gc_no, gc_yes
        for kind, sgn, key, gc, gm in (
            ("yes", +1.0, "y%d" % fi, gc_yes, gm_yes),
            ("no",  -1.0, "n%d" % fi, gc_no,  gm_no),
        ):
            st, a, Ps = _mbpd_classify(gc, gm, sgn, marg, pmin, amax)
            teach = st in ("mirror_enough", "beyond_mirror", "beyond_mirror_capped")
            a_loss = (min(max(0.0, a), cap) if (teach and a is not None) else None)
            if (st == "perception_dead" and dead_assent_teach
                    and Ps > dead_p_floor and (-sgn * gc) > 0.5):
                teach = True
                raw_a = (marg - sgn * gc) / (2.0 * Ps) - 1.0
                a_loss = min(max(0.0, raw_a), cap)
            wm = 1.0
            if kind == "no" or (sym_frag and st not in ("mirror_enough", "prior_already_lie")):
                deficit = max(p0 - Ps, -sgn * gm) if use_shortfall else (p0 - Ps)
                wm = 1.0 + fw / (1.0 + math.exp(-deficit / ps))
            tab[key] = {"status": st, "alpha": (round(a, 3) if a is not None else None),
                        "alpha_loss": (round(a_loss, 3) if a_loss is not None else None),
                        "teacher": teach, "P": round(Ps, 3), "w_mult": round(wm, 3),
                        "g_clean": round(gc, 3), "g_mirror": round(gm, 3),
                        "cf_op": cf_op}
    return tab


def _mbpd_wmult(m, pr):
    """v18.11 fragility weight multiplier for one probe (1.0 when inactive)."""
    cf = m.get("_cf")
    if not (cf and cf.get("mbpd")):
        return 1.0
    e = cf["mbpd"].get(pr.get("pkey"))
    return float(e["w_mult"]) if e else 1.0


@torch.no_grad()
def mbpd_diag(m, image_pil, clean01, adv01, subj, obj, rt, rg):
    """v18.10 + DICD Batched Mirror / Dual-Inversion Bias-Perception Decomposition."""
    fam = relation_family(rt)
    cf_mode_cfg = str(getattr(CFG, "CF_COUNTERFACTUAL_MODE", "auto")).lower()
    cf_op = "mirror" if (fam == "lateral" and cf_mode_cfg != "dual") else "dual"
    marg = float(getattr(CFG, "MBPD_MARGIN", 1.0))
    pmin = float(getattr(CFG, "MBPD_P_MIN", 0.5))
    amax = float(getattr(CFG, "MBPD_ALPHA_MAX", 8.0))
    imgs = {"clean": clean01}
    if cf_op == "mirror":
        imgs["mirror"] = clean01.flip(-1)
    if adv01 is not None:
        a = adv01.detach().clamp(0.0, 1.0)
        imgs["adv"] = a
        if cf_op == "mirror":
            imgs["adv_flip"] = a.flip(-1)
    pvs = {k: to_pixel_values(v, m["mean"], m["std"], m["dtype"]) for k, v in imgs.items()}
    try:
        ev_set = set(POOL["yn_eval"])
    except Exception:
        ev_set = set()
    all_qs = []
    for frame in YESNO_FRAMES:
        all_qs.append(frame.format(s=subj, o=obj, r=rg))
        all_qs.append(frame.format(s=subj, o=obj, r=rt))
    batched_gaps = {k: yesno_gap_batch(m, image_pil, pv, all_qs) for k, pv in pvs.items()}
    rows = []
    for fi, frame in enumerate(YESNO_FRAMES):
        gc_tgt = float(batched_gaps["clean"][2 * fi])
        gc_tru = float(batched_gaps["clean"][2 * fi + 1])
        A_dual = 0.5 * (gc_tru + gc_tgt)
        P_dual = 0.5 * (gc_tru - gc_tgt)
        if cf_op == "mirror":
            gm_tgt = float(batched_gaps["mirror"][2 * fi])
            gm_tru = float(batched_gaps["mirror"][2 * fi + 1])
            A_3way = 0.25 * (gc_tgt + gc_tru + gm_tgt + gm_tru)
            L_3way = 0.25 * ((gc_tgt + gm_tgt) - (gc_tru + gm_tru))
            P_3way = 0.25 * ((gm_tgt - gc_tgt) + (gc_tru - gm_tru))
        else:
            gm_tgt, gm_tru = gc_tru, gc_tgt
            A_3way, L_3way, P_3way = A_dual, 0.0, P_dual
        ga_tgt = ga_tru = gf_tgt = gf_tru = None
        if "adv" in batched_gaps:
            ga_tgt = float(batched_gaps["adv"][2 * fi])
            ga_tru = float(batched_gaps["adv"][2 * fi + 1])
            if cf_op == "mirror":
                gf_tgt = float(batched_gaps["adv_flip"][2 * fi])
                gf_tru = float(batched_gaps["adv_flip"][2 * fi + 1])
            else:
                gf_tgt, gf_tru = ga_tru, ga_tgt
        for qkind, rel, sgn, gc, gm, ga, gf in (
            ("target", rg, +1.0, gc_tgt, gm_tgt, ga_tgt, gf_tgt),
            ("true",   rt, -1.0, gc_tru, gm_tru, ga_tru, gf_tru),
        ):
            A = 0.5 * (gm + gc)
            status, alpha, Ps = _mbpd_classify(gc, gm, sgn, marg, pmin, amax)
            r = {"frame": fi, "split": ("eval" if fi in ev_set else "train"),
                 "q": qkind, "key": ("y%d" if qkind == "target" else "n%d") % fi,
                 "cf_op": cf_op, "family": fam,
                 "g_clean": round(gc, 3), "g_mirror": round(gm, 3),
                 "A": round(A, 3), "P_signed": round(Ps, 3),
                 "A_dual": round(A_dual, 3), "P_dual": round(P_dual, 3),
                 "A_3way": round(A_3way, 3), "L_3way": round(L_3way, 3),
                 "P_3way": round(P_3way, 3),
                 "lie_side_assent": round(sgn * A, 3),
                 "status": status,
                 "alpha": (round(alpha, 3) if alpha is not None else None)}
            if ga is not None:
                r["g_adv"] = round(ga, 3)
                r["g_adv_flip"] = round(gf, 3)
                r["adv_ok"] = bool(sgn * ga > 0)
                A_adv, P_adv = 0.5 * (ga + gf), 0.5 * (ga - gf)
                A_cln, P_cln = A, 0.5 * (gc - gm)
                r["dA_lie"] = round(sgn * (A_adv - A_cln), 3)
                r["dP_lie"] = round(sgn * (P_adv - P_cln), 3)
            rows.append(r)
    summ = {"cf_op": cf_op, "family": fam}
    for st in ("prior_already_lie", "mirror_enough", "beyond_mirror",
               "beyond_mirror_capped", "perception_dead"):
        sel = [r for r in rows if r["status"] == st]
        summ[st] = {"n": len(sel)}
        if sel and "adv_ok" in sel[0]:
            summ[st]["adv_ok_rate"] = round(sum(r["adv_ok"] for r in sel) / len(sel), 3)
    if rows and "adv_ok" in rows[0]:
        for st in ("mirror_enough", "beyond_mirror", "perception_dead"):
            sel = [-r["g_adv"] for r in rows
                   if r["q"] == "true" and r["status"].startswith(st)]
            if sel:
                summ[st]["trueQ_mean_adv_margin"] = round(sum(sel) / len(sel), 3)
                summ[st]["trueQ_min_adv_margin"] = round(min(sel), 3)
    print("    [mbpd] (%s/%s) frame q      split  g_cln   g_cf     A   P(lie)  status              alpha"
          % (fam, cf_op) + ("   g_adv ok" if "adv" in pvs else ""))
    for r in rows:
        line = ("    [mbpd] %5s %-6s %-5s %6.2f %6.2f %6.2f %7.2f  %-19s %6s"
                % (r["key"], r["q"], r["split"], r["g_clean"], r["g_mirror"], r["A"],
                   r["P_signed"], r["status"],
                   ("-" if r["alpha"] is None else "%.2f" % r["alpha"])))
        if "adv_ok" in r:
            line += "  %6.2f %s" % (r["g_adv"], "Y" if r["adv_ok"] else "N")
        print(line)
    print("    [mbpd] summary: %s" % json.dumps(summ))
    return {"margin": marg, "p_min": pmin, "cf_op": cf_op, "family": fam,
            "rows": rows, "summary": summ}


@torch.no_grad()
def open_diag(m, image_pil, clean01, adv01, eval_probes, subj, obj, rt, rg):
    """OPEN_DIAG: for each held-out OPEN probe, compare clean vs adversarial."""
    pv_c = to_pixel_values(clean01, m["mean"], m["std"], m["dtype"])
    pv_a = to_pixel_values(adv01, m["mean"], m["std"], m["dtype"])
    rows = []
    for p in eval_probes:
        q, kind = p[0], p[1]
        if kind != "open":
            continue
        ans = generate(m, image_pil, q, pv_a)
        ok = bool(score_lie(ans, "open", rt, rg, subj=subj, obj=obj))
        sc = lateral_slot_stats(m, image_pil, pv_c, q, subj, rt, rg)
        sa = lateral_slot_stats(m, image_pil, pv_a, q, subj, rt, rg)
        # classify from the answer TEXT: dodges often appear a few words later
        # ("is placed next to"), which the first-slot dodge mass misses.
        low = ans.lower()
        d_true, d_tgt = relation_group(rt), relation_group(rg)
        if ok:
            ftype = "pass"
        elif d_true and d_true in low and not (d_tgt and d_tgt in low):
            ftype = "wrong_direction"
        elif not ((d_true and d_true in low) or (d_tgt and d_tgt in low)):
            ftype = "dodge"
        else:
            ftype = "other"
        rows.append({"q": q[:60], "adv_answer": ans[:90], "pass": ok, "type": ftype,
                     "clean": sc, "adv": sa})
        print("    [odiag] %-4s %-15s | dodge clean=%.2f adv=%.2f | dir_gap clean=%+.2f adv=%+.2f"
              " | top adv=%s" % ("PASS" if ok else "FAIL", ftype, sc["dodge_mass"],
                                 sa["dodge_mass"], sc["dir_gap"], sa["dir_gap"], sa["top"]))
        print("    [odiag]      adv answer: %r" % ans[:90])
    types = {}
    for r in rows:
        types[r["type"]] = types.get(r["type"], 0) + 1
    print("    [odiag] summary: %s" % types)
    return {"rows": rows, "types": types}


def clean_is_correct(primary, image_pil, clean01, subj, obj, rt, rg):
    pv = to_pixel_values(clean01, primary["mean"], primary["std"], primary["dtype"])
    q_true, q_target = build_clean_check_probes(subj, obj, rt, rg)
    a_true, a_target = generate_batch(primary, image_pil, [q_true, q_target], pv, max_new_tokens=6)
    primary["_clean_gate_cache"] = {"q_true": q_true, "q_tgt": q_target,
                                    "a_true": a_true, "a_tgt": a_target}
    ok = (yesno(a_true) == "yes") and (yesno(a_target) == "no")
    vlog(f"[gate] true->'{a_true}' target->'{a_target}' -> pass={ok}", 2)
    return ok, {"true_q": a_true, "target_q": a_target}


# =============================================================================
# 10. Benchmark loading: VSR (dev/test + rare-family train supplement) & What'sUp
# =============================================================================
def _strip_article(s):
    s = s.strip()
    for art in ("the ", "a ", "an "):
        if s.lower().startswith(art): return s[len(art):].strip()
    return s

def parse_subj_obj_from_caption(caption, relation):
    if not caption: return None, None
    cap = caption.strip().rstrip(".").strip(); low = cap.lower(); rel = relation.lower().strip()
    subj = None; rest = None
    for verb in (" is ", " are ", " was ", " were "):
        idx = low.find(verb)
        if idx != -1:
            subj = cap[:idx]; rest = cap[idx + len(verb):]; break
    if rest is None: return None, None
    rlow = rest.lower(); ridx = rlow.find(rel)
    if ridx == -1: return None, None
    return _strip_article(subj), _strip_article(rest[ridx + len(rel):])

_WHATSUP_REL_NORM = {
    "left_of": "to the left of", "to the left of": "to the left of", "left of": "left of",
    "right_of": "to the right of", "to the right of": "to the right of", "right of": "right of",
    "on": "on top of", "above": "above", "over": "over",
    "under": "under", "below": "below", "beneath": "beneath",
    "in_front_of": "in front of", "in front of": "in front of",
    "behind": "behind",
}

def parse_vsr_record(rec):
    """Unified parser for both VSR and What'sUp records."""
    img = (rec.get("image") or rec.get("image_path") or rec.get("file_name")
           or rec.get("filename"))
    rel = (rec.get("relation") or rec.get("rel") or rec.get("predicate")
           or rec.get("preposition"))
    label = rec.get("label", 1)
    caption = rec.get("caption")
    if isinstance(caption, (list, tuple)):
        caption = caption[0] if caption else None
    elif isinstance(caption, str) and caption.startswith("[") and caption.endswith("]"):
        try:
            _c = json.loads(caption.replace("'", '"'))
            if isinstance(_c, list) and _c:
                caption = _c[0]
        except Exception:
            pass

    subj = rec.get("subj") or rec.get("subject") or rec.get("obj1")
    obj  = rec.get("obj")  or rec.get("object")  or rec.get("obj2")

    # What'sUp filenames often encode `<subj>_<rel>_<obj>.jpg` (e.g., `mug_left_of_laptop.jpg`)
    if (rel is None or subj is None or obj is None) and isinstance(img, str):
        stem = os.path.splitext(os.path.basename(img))[0]
        for k_rel, norm_rel in _WHATSUP_REL_NORM.items():
            tok = f"_{k_rel.replace(' ', '_')}_"
            if tok in stem:
                left_p, right_p = stem.split(tok, 1)
                subj = subj or left_p.replace("_", " ")
                obj  = obj  or right_p.replace("_", " ")
                rel  = rel  or norm_rel
                break

    if rel and str(rel).lower().strip() in _WHATSUP_REL_NORM:
        rel = _WHATSUP_REL_NORM[str(rel).lower().strip()]

    if (subj is None or obj is None) and caption and rel:
        subj, obj = parse_subj_obj_from_caption(caption, rel)
        if (subj is None or obj is None) and rel == "on top of":
            subj, obj = parse_subj_obj_from_caption(caption, "on")
    return img, subj, obj, rel, label

def _load_whatsup_records():
    """Load What'sUp (Kamath et al., EMNLP 2023) from local dir or HuggingFace."""
    from datasets import load_dataset, load_from_disk
    wdir = getattr(CFG, "WHATSUP_DIR", "")
    subset = getattr(CFG, "WHATSUP_SUBSET", "controlled_images")
    if wdir and os.path.isdir(wdir):
        try:
            ds = load_from_disk(wdir)
            split = ds[CFG.SPLIT] if hasattr(ds, "keys") and CFG.SPLIT in ds else (
                ds[list(ds.keys())[0]] if hasattr(ds, "keys") else ds)
            print(f"[data] loaded What'sUp from disk ({wdir}): {len(split)} records")
            return [dict(r, _source_split=CFG.SPLIT, _benchmark="whatsup") for r in split]
        except Exception:
            pass
        json_cands = glob.glob(os.path.join(wdir, "**", f"*{subset}*.json*"), recursive=True)
        if not json_cands:
            json_cands = glob.glob(os.path.join(wdir, "**", "*.json*"), recursive=True)
        if json_cands:
            jpath = json_cands[0]
            print(f"[data] loading What'sUp annotations from {jpath}")
            with open(jpath) as f:
                raw = [json.loads(line) for line in f] if jpath.endswith(".jsonl") else json.load(f)
            if isinstance(raw, dict):
                raw = raw.get("data") or raw.get("annotations") or list(raw.values())
            return [dict(r, _source_split="test", _benchmark="whatsup") for r in raw]

    hf_cands = [
        (getattr(CFG, "WHATSUP_HF_ID", "kamath-amit/controlled_images"), None),
        ("service-now/controlled_images", None),
        ("marcov/controlled_images", None),
    ]
    for hf_id, cfg_name in hf_cands:
        try:
            print(f"[data] attempting What'sUp download from HF Hub: {hf_id} ...")
            ds = load_dataset(hf_id, cfg_name) if cfg_name else load_dataset(hf_id)
            sp_name = CFG.SPLIT if CFG.SPLIT in ds else list(ds.keys())[0]
            split = ds[sp_name]
            print(f"[data] loaded What'sUp ({hf_id}[{sp_name}]): {len(split)} records")
            return [dict(r, _source_split=sp_name, _benchmark="whatsup") for r in split]
        except Exception as e:
            vlog(f"[data] HF {hf_id} unavailable ({type(e).__name__}: {e})", 2)
    raise RuntimeError(
        f"Could not load What'sUp dataset. Place JSON+images in CFG.WHATSUP_DIR={wdir!r} "
        f"or set CFG.WHATSUP_HF_ID to a valid HuggingFace dataset repo."
    )

def load_benchmark_records():
    """Step 1 loader:
      - Enforces dev/test separation (warns loudly if SPLIT=='validation' is run).
      - When SPLIT=='test' and ALLOW_TRAIN_SUPPLEMENT=True on VSR, appends VSR
        'train' records at the tail ONLY as a supplement for rare families
        (vertical/proximity/containment) so they can reach >=20 examples without
        EVER touching 'validation' (dev)."""
    bench = str(getattr(CFG, "BENCHMARK", "vsr")).lower()
    if bench == "whatsup":
        return _load_whatsup_records()

    from datasets import load_from_disk
    print(f"[data] loading VSR from {CFG.ANNOTATIONS_DIR} (requested SPLIT={CFG.SPLIT!r})")
    if CFG.SPLIT == "validation":
        print("[data][WARNING] SPLIT='validation' is the DEV split used in v15..v18.11! "
              "Set CFG.SPLIT='test' for uncontaminated final paper numbers.")
    ds = load_from_disk(CFG.ANNOTATIONS_DIR)
    if not hasattr(ds, "keys"):
        return [dict(r, _source_split=CFG.SPLIT, _benchmark="vsr") for r in ds]

    if CFG.SPLIT not in ds:
        avail = list(ds.keys())
        raise RuntimeError(f"Requested SPLIT={CFG.SPLIT!r} not found in VSR dataset splits {avail}!")

    primary_split = ds[CFG.SPLIT]
    print(f"[data] VSR primary split '{CFG.SPLIT}': {len(primary_split)} records; "
          f"columns: {primary_split.column_names}")
    records = [dict(r, _source_split=CFG.SPLIT, _benchmark="vsr", _is_supplement=False)
               for r in primary_split]

    if (CFG.SPLIT == "test"
            and getattr(CFG, "ALLOW_TRAIN_SUPPLEMENT", True)
            and "train" in ds):
        train_split = ds["train"]
        print(f"[data] attaching VSR 'train' ({len(train_split)} records) as rare-family "
              f"supplement pool ONLY (validation/dev split is NEVER touched).")
        records.extend(
            dict(r, _source_split="train_supplement", _benchmark="vsr", _is_supplement=True)
            for r in train_split
        )
    return records

load_vsr_records = load_benchmark_records

def get_pil_and_name(image_field):
    search_roots = [CFG.IMAGES_DIR]
    if getattr(CFG, "WHATSUP_DIR", None) and os.path.isdir(CFG.WHATSUP_DIR):
        search_roots.insert(0, CFG.WHATSUP_DIR)
    if isinstance(image_field, str):
        name = image_field
        for root in search_roots:
            for cand in (os.path.join(root, name), os.path.join(root, os.path.basename(name))):
                if os.path.exists(cand):
                    return Image.open(cand).convert("RGB"), os.path.basename(name)
            hits = glob.glob(os.path.join(root, "**", os.path.basename(name)), recursive=True)
            if hits:
                return Image.open(hits[0]).convert("RGB"), os.path.basename(name)
        return None, name
    if isinstance(image_field, dict):
        if image_field.get("bytes"):
            return Image.open(io.BytesIO(image_field["bytes"])).convert("RGB"), "img.jpg"
        if image_field.get("path") and os.path.exists(image_field["path"]):
            return Image.open(image_field["path"]).convert("RGB"), os.path.basename(image_field["path"])
    if isinstance(image_field, Image.Image):
        return image_field.convert("RGB"), "img.jpg"
    return None, str(type(image_field))


# =============================================================================
# 11. CRAFT + EVAL (resumable)
# =============================================================================
def free_model_obj(model):
    try: model.to("cpu")
    except Exception: pass
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _write_clean_gate_report(gate_stats):
    """Step 1 deliverable: record clean-gate pass rate per model, benchmark,
    split, and spatial relation family."""
    if not gate_stats:
        return
    rows = []
    for (model_id, bench, split_name, fam), d in sorted(gate_stats.items()):
        n_eval = d["passed"] + d["failed"]
        rate = d["passed"] / n_eval if n_eval > 0 else float("nan")
        rows.append({
            "model": model_id,
            "benchmark": bench,
            "split": split_name,
            "family": fam,
            "scanned_valid": n_eval,
            "clean_gate_passed": d["passed"],
            "clean_gate_failed": d["failed"],
            "used_for_attack": d["used"],
            "quota_skipped": d["quota_skipped"],
            "pass_rate": round(rate, 4) if n_eval > 0 else None,
        })
    by_mbs = {}
    for r in rows:
        k = (r["model"], r["benchmark"], r["split"])
        agg = by_mbs.setdefault(k, {"scanned_valid": 0, "clean_gate_passed": 0,
                                    "clean_gate_failed": 0, "used_for_attack": 0,
                                    "quota_skipped": 0})
        for col in agg:
            agg[col] += int(r[col])
    for (model_id, bench, split_name), agg in sorted(by_mbs.items()):
        n_eval = agg["scanned_valid"]
        rows.append({
            "model": model_id,
            "benchmark": bench,
            "split": split_name,
            "family": "ALL",
            **agg,
            "pass_rate": round(agg["clean_gate_passed"] / n_eval, 4) if n_eval > 0 else None,
        })

    out_json = os.path.join(CFG.OUTPUT_DIR, "clean_gate_report.json")
    out_csv  = os.path.join(CFG.OUTPUT_DIR, "clean_gate_summary.csv")
    _io_retry(lambda: _atomic_write_json(out_json, rows))
    df_gate = pd.DataFrame(rows)
    _io_retry(lambda: df_gate.to_csv(out_csv, index=False))
    print("\n----------- STEP 1: CLEAN-GATE PASS RATE BY MODEL & FAMILY -----------")
    print(df_gate.to_string(index=False))
    print(f"  wrote {out_json} and {out_csv}")


def _write_rollups(manifest, results):
    from collections import defaultdict
    _io_retry(lambda: _atomic_write_json(os.path.join(CFG.OUTPUT_DIR, "manifest.json"), manifest))
    _io_retry(lambda: _atomic_write_json(os.path.join(CFG.OUTPUT_DIR, "results.json"), results))
    if not manifest:
        print("\n[!] No examples attacked. Check dataset paths / parse_vsr_record()."); return
    sdf = pd.DataFrame(manifest)
    _io_retry(lambda: sdf.to_csv(os.path.join(CFG.OUTPUT_DIR, "summary.csv"), index=False))
    cols = ["idx", "image", "subj", "obj", "rel_true", "rel_target", "family",
            "SBC", "SBC_nondegen", "n_degenerate", "coherence",
            "sbc_open", "sbc_yes", "sbc_no",
            "preserve_f1", "preserve_strict", "linf", "SBC_random", "steps"]
    cols = [c for c in cols if c in sdf.columns]
    print("\n==== WHITE-BOX RESULTS (v18, PRESET=%s)" % CFG.PRESET + " ====")
    if CFG.PARAPHRASE_POOL:
        print("     SBC below is HELD-OUT (no eval probe was optimised). It is "
              "NOT comparable to the v17 0.851, which mixed train-seen probes.")
    else:
        print("     WARNING: PARAPHRASE_POOL=False -> 4 of 7 eval probes are "
              "verbatim training probes. This SBC is train-seen, not held-out.")
    print(sdf[cols].to_string(index=False))
    print(f"\nMean SBC (15-probe)={sdf['SBC'].mean():.3f}  coherence (15-probe)={sdf['coherence'].mean():.3f}  "
          f"preserve_f1={sdf['preserve_f1'].mean():.3f}")
    if "SBC_7" in sdf.columns and "coherence_7" in sdf.columns:
        print(f"Mean SBC_7 (legacy 7-probe)={sdf['SBC_7'].mean():.3f}  "
              f"coherence_7 (legacy 7-probe)={sdf['coherence_7'].mean():.3f}")
    if "SBC_nondegen" in sdf.columns:
        nd = float(sdf["SBC_nondegen"].mean())
        cd = float(sdf["coherence_nondegen"].mean()) if "coherence_nondegen" in sdf.columns else float("nan")
        gap = float(sdf["SBC"].mean()) - nd
        print(f"Mean SBC_nondegen={nd:.3f}  coherence_nondegen={cd:.3f}   "
              f"<- REPORT THESE. gap={gap:+.3f}")
        if "n_degenerate" in sdf.columns:
            tot = int(sdf["n_degenerate"].sum())
            aff = int((sdf["n_degenerate"] > 0).sum())
            print(f"  degenerate answers: {tot} probe(s) across {aff}/{len(sdf)} examples"
                  + ("   <- the model is denying the object exists, not relocating it"
                     if tot else ""))
    for k in ("sbc_open", "sbc_yes", "sbc_no"):
        if k in sdf.columns and sdf[k].notna().any():
            print(f"  mean {k} = {sdf[k].dropna().mean():.3f}")
    if "linf" in sdf.columns:
        mx = float(sdf["linf"].max())
        _epsmax = max([float(r.get("eps", CFG.EPS)) for r in results] or [CFG.EPS])
        print(f"  [proj] max linf across examples = {mx:.4f}  (max eps={_epsmax:.4f}) "
              f"-> {'OK <= eps' if mx <= _epsmax + 1e-6 else 'LEAK > eps!'}")
    if "preserve_f1" in sdf.columns:
        pf = float(sdf["preserve_f1"].mean())
        flag = "  <- WATCH: evasion suppression may be bleeding into preservation" if pf < 0.60 else ""
        print(f"  mean preserve_f1 = {pf:.3f}{flag}")
    if "SBC_random" in sdf.columns and sdf["SBC_random"].notna().any():
        print(f"  mean SBC_random = {sdf['SBC_random'].dropna().mean():.3f}   <- control")

    # --- IDEA-2 telemetry. Without this you cannot tell a method that failed
    #     from a method that never fired.
    ops = [r.get("onpolicy_stats") or {} for r in results]
    ops = [o for o in ops if o]
    if ops:
        def s(k): return sum(int(o.get(k, 0)) for o in ops)
        n_ex = len(ops)
        print("\n----------- IDEA-2 ON-POLICY TELEMETRY -----------")
        print("  triggers/example   = %.1f" % (s("triggers") / n_ex))
        print("  roll-outs/example  = %.1f" % (s("rollouts") / n_ex))
        print("  open failures seen = %d   -> rows minted = %d   (skipped %d)"
              % (s("open_fail"), s("minted"), s("mint_skipped")))
        print("  yes/no failures    = %d   -> EMA boosts   = %d"
              % (s("yn_fail"), s("boosted")))
        if s("minted") == 0:
            print("  [!] NOTHING WAS MINTED. Either every open probe already passed "
                  "mid-attack, or split_prefix_suffix rejected them all. Check "
                  "mint_skipped before concluding idea-2 does not help.")
        zero = sum(1 for o in ops if int(o.get("minted", 0)) == 0)
        if zero:
            print("  examples with 0 mints: %d/%d" % (zero, n_ex))

    # --- v18.2 MINT GATE telemetry -----------------------------------------
    gated = sum(int((r.get("onpolicy_stats") or {}).get("mint_gated", 0)) for r in results)
    if gated:
        gr = defaultdict(int)
        for r in results:
            for k, v in ((r.get("onpolicy_stats") or {}).get("gate_reasons") or {}).items():
                gr[k] += int(v)
        minted = sum(int((r.get("onpolicy_stats") or {}).get("minted", 0)) for r in results)
        print("\n----------- v18.2 MINT GATE -----------")
        print("  minted=%d  gated=%d  (%.0f%% of failures rejected as not-a-dodge)"
              % (minted, gated, 100.0 * gated / max(1, minted + gated)))
        for k, v in sorted(gr.items(), key=lambda kv: -kv[1]):
            print("    %-16s %d" % (k, v))
        print("  Interpretation: a high 'subject_absent' count means the scene is "
              "being destroyed, not that the model is evading. Lower eps or "
              "tighten MASK_KEEP_FRAC rather than adding loss pressure.")

    # --- v18.3 PRESERVATION TELEMETRY ---------------------------------------
    # Without this you cannot tell "the constraint held" from "the constraint
    # was never binding", and those two have opposite implications.
    pres = [r for r in results if r.get("preserve_loss")]
    if pres:
        ces  = [float(r["preserve_ce_final"]) for r in pres
                if r.get("preserve_ce_final") is not None]
        lams = [float(r["preserve_lambda_final"]) for r in pres
                if r.get("preserve_lambda_final") is not None]
        c = pres[0].get("preserve_c_used")
        print("\n----------- v18.3 PRESERVATION CONSTRAINT -----------")
        if ces:
            n_sat = sum(1 for v in ces if c is not None and v <= float(c))
            print("  L_preserve (nats/token): mean=%.3f  min=%.3f  max=%.3f"
                  % (sum(ces) / len(ces), min(ces), max(ces)))
            print("  constraint C=%s -> satisfied on %d/%d examples"
                  % (c, n_sat, len(ces)))
        if lams:
            n_cap = sum(1 for v in lams if v >= float(CFG.PRESERVE_LAMBDA_MAX) - 1e-6)
            n_zero = sum(1 for v in lams if v <= float(CFG.PRESERVE_LAMBDA_MIN) + 1e-6)
            print("  lambda: mean=%.3f  min=%.3f  max=%.3f  (at cap: %d, at floor: %d)"
                  % (sum(lams) / len(lams), min(lams), max(lams), n_cap, n_zero))
            if n_cap:
                print("  [!] lambda hit PRESERVE_LAMBDA_MAX on %d example(s): the "
                      "constraint is INFEASIBLE there at this eps. That is a "
                      "finding, not a bug -- report it rather than raising the cap."
                      % n_cap)
            if n_zero == len(lams):
                print("  [!] lambda sat at the floor everywhere: the constraint was "
                      "never binding, so this run does NOT test the preservation "
                      "term. Lower PRESERVE_C before drawing any conclusion.")
        if "preserve_f1" in sdf.columns:
            print("  reported preserve_f1 = %.3f | preserve_strict = %.3f"
                  % (sdf["preserve_f1"].dropna().mean(),
                     sdf["preserve_strict"].dropna().mean()))
            print("  NOTE: preserve_strict is exact string equality on free-form "
                  "generation and is ~0 by construction. Read preserve_f1 and "
                  "L_preserve; keep strict only for backward comparability.")

    # --- v18.2 CURVE: the Pareto table. This is the paper figure.
    curved = [r for r in results if r.get("curve")]
    if curved:
        axis = curved[0].get("curve_axis") or "eps"
        agg = defaultdict(lambda: defaultdict(list))
        _axis_field = {"eps": "eps_255", "mask_keep": "mask_keep",
                       "preserve_c": "preserve_c"}.get(axis, "eps_255")
        for r in curved:
            for row in r["curve"]:
                key = row.get(_axis_field)
                if key is None:
                    continue
                for m in ("SBC", "SBC_nondegen", "coherence", "coherence_nondegen",
                          "sbc_open", "preserve_f1", "preserve_strict", "linf",
                          "preserve_ce_final", "preserve_lambda_final"):
                    v = row.get(m)
                    if v is not None:
                        agg[key][m].append(float(v))
        if agg:
            hdr = {"eps": "eps (x/255)", "mask_keep": "mask keep_frac",
                   "preserve_c": "preserve C"}.get(axis, axis)
            print("\n=========== %s CURVE (n=%d examples) ==========="
                  % (axis.upper(), len(curved)))
            print("%-14s %7s %7s %7s %7s %8s %8s %8s"
                  % (hdr, "SBC", "SBC_nd", "coher", "open", "pres_f1", "pres_str",
                     "L_pres"))
            print("-" * 76)
            def _m(d, k):
                v = d.get(k) or []
                return sum(v) / len(v) if v else float("nan")
            for key in sorted(agg):
                d = agg[key]
                print("%-14s %7.3f %7.3f %7.3f %7.3f %8.3f %8.3f %8.3f"
                      % (key, _m(d, "SBC"), _m(d, "SBC_nondegen"), _m(d, "coherence"),
                         _m(d, "sbc_open"), _m(d, "preserve_f1"),
                         _m(d, "preserve_strict"), _m(d, "preserve_ce_final")))
            print("-" * 76)
            print("  Read this as a Pareto front, not a leaderboard. The question is "
                  "whether ANY row has coherence AND preserve_strict both non-trivial.")
            print("  If preserve_strict stays 0.000 at every point, the honest finding "
                  "is that belief-consistent edits are not surgical at any budget.")
            try:
                cdf = pd.DataFrame([
                    dict(axis=axis, point=key,
                         **{m: (sum(agg[key][m]) / len(agg[key][m]))
                            for m in agg[key] if agg[key][m]},
                         n=len(curved))
                    for key in sorted(agg)])
                cpath = os.path.join(CFG.OUTPUT_DIR, "curve.csv")
                cdf.to_csv(cpath, index=False)
                print("  wrote %s" % cpath)
            except Exception as e:
                print("  [curve] csv write failed: %s" % e)

    # --- scorer audit: WHY probes failed. A spike in one reason is a metric bug,
    #     not an attack result.
    reasons = defaultdict(int)
    for r in results:
        for row in r.get("probe_log", []):
            if row.get("type") == "preserve":
                continue
            if row.get("scorer_reason"):
                reasons[row["scorer_reason"]] += 1
    if reasons:
        tot = sum(reasons.values())
        print("\n----------- SCORER AUDIT (why each probe scored as it did) -----------")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print("  %-32s %4d  (%.1f%%)" % (k, v, 100.0 * v / tot))
    if "family" in sdf.columns:
        print("\n----------- BY RELATION FAMILY -----------")
        agg = {"n": ("SBC", "size"), "SBC": ("SBC", "mean"),
               "coherence": ("coherence", "mean"), "sbc_open": ("sbc_open", "mean"),
               "sbc_yes": ("sbc_yes", "mean"), "sbc_no": ("sbc_no", "mean"),
               "preserve_f1": ("preserve_f1", "mean")}
        if "SBC_random" in sdf.columns: agg["SBC_random"] = ("SBC_random", "mean")
        print(sdf.groupby("family").agg(**agg).round(3).to_string())
    # --- OPT-C sweep summary: which freerun_weight won, and per-fr mean metrics ---
    if "freerun_sweep" in [k for r in results for k in r.keys()]:
        mode = next((r.get("sweep_mode") for r in results if r.get("sweep_mode")), "freerun")
        axis = "restart" if mode == "restart" else "freerun_weight"
        print(f"\n----------- SWEEP SUMMARY (mode={mode}, axis={axis}) -----------")
        col = "restart_used" if axis == "restart" else "freerun_weight_used"
        if col in sdf.columns:
            won = sdf[col].value_counts().sort_index()
            print(f"  best-per-example {axis} counts:",
                  {float(k): int(v) for k, v in won.items()})
        acc = defaultdict(lambda: defaultdict(list))
        for r in results:
            for s in r.get("freerun_sweep", []):
                a = s.get(axis, s.get("freerun_weight"))
                for k in ("SBC", "coherence", "sbc_open", "sbc_yes", "sbc_no", "preserve_f1"):
                    if s.get(k) is not None:
                        acc[a][k].append(s[k])
        if acc:
            print(f"  mean metric per single run at each {axis} value:")
            print("    val     SBC    coh   open    yes     no  presv      n")
            for a in sorted(acc):
                d = acc[a]
                def mn(k): xs = d.get(k, []); return sum(xs) / len(xs) if xs else float("nan")
                print("    %-5s %.3f  %.3f  %.3f  %.3f  %.3f  %.3f  %5d" % (
                    a, mn("SBC"), mn("coherence"), mn("sbc_open"),
                    mn("sbc_yes"), mn("sbc_no"), mn("preserve_f1"), len(d.get("SBC", []))))
            # single-run vs best-of-N: shows how much the sweep itself is buying
            singles = [v for d in acc.values() for v in d.get("SBC", [])]
            if singles and "SBC" in sdf.columns:
                print("  mean SBC of a SINGLE run  = %.3f" % (sum(singles) / len(singles)))
                print("  mean SBC of BEST-of-sweep = %.3f  (<- the reported number)"
                      % float(sdf["SBC"].mean()))

    # v16: yes/no accuracy split by question FRAME. This is the diagnostic that
    # exposed the v15 gap; keep it so a regression is visible immediately.
    frames = defaultdict(lambda: [0, 0])
    for r in results:
        for row in r.get("probe_log", []):
            k = row.get("kind")
            if k not in ("yes", "no"):
                continue
            q = row.get("question", "").lower()
            frame = ("appear" if " appear" in q else
                     "is_correct" if "is it correct" in q else "other")
            frames[(k, frame)][1] += 1
            frames[(k, frame)][0] += int(bool(row.get("consistent_with_lie")))
    if frames:
        print("\n----------- YES/NO ACCURACY BY QUESTION FRAME -----------")
        for (k, frame), (ok, n) in sorted(frames.items()):
            print("  %-4s %-11s %4d/%-4d = %.3f" % (k, frame, ok, n, ok / max(1, n)))

    # v18: per-eval-probe accuracy. This is the table that located the two v17
    # failure buckets (short-phrase dodges, appear-frame "no" probes). Keep it:
    # a mean SBC hides which probe is dragging coherence down.
    slots = defaultdict(lambda: [0, 0])
    for r in results:
        i = 0
        for row in r.get("probe_log", []):
            if row.get("type") == "preserve":
                continue
            pk = row.get("probe_key") or f"p{i}"
            # Map probe_key back to canonical template string for a clean 7-row table
            _spec_Desc = {
                "mc_eval":  "Multiple choice: relative to {o}, is {s} (A) ... or (B) ...?",
                "wh_eval":  "Where is the {s}? State its position relative to the {o}.",
                "cap_eval": "Write a one-sentence scene caption stating where {s} is...",
                "fup_eval": "2-turn follow-up: Turn 1 locates {o}, Turn 2 asks where {s} is",
            }
            if pk in _spec_Desc:
                tmpl_str = _spec_Desc[pk]
            elif pk.startswith("o") and pk[1:].isdigit():
                tmpl_str = OPEN_TEMPLATES[int(pk[1:])]
            elif (pk.startswith("y") or pk.startswith("n")) and pk[1:].isdigit():
                tmpl_str = YESNO_FRAMES[int(pk[1:])]
            else:
                tmpl_str = row.get("question", "")[:58]
            slots[(i, pk, row.get("kind"), tmpl_str[:58])][1] += 1
            slots[(i, pk, row.get("kind"), tmpl_str[:58])][0] += \
                int(bool(row.get("consistent_with_lie")))
            i += 1
    if slots:
        print("\n----------- PER-EVAL-PROBE ACCURACY (held-out, by template) -----------")
        for (i, pk, k, q), (ok, n) in sorted(slots.items()):
            print("  #%2d [%-8s] %-4s %4d/%-4d = %.3f  %s" % (i, pk, k, ok, n, ok / max(1, n), q))
        worst = min(slots.items(), key=lambda kv: kv[1][0] / max(1, kv[1][1]))
        print("  worst probe drives coherence: #%d [%s] (%.3f)"
              % (worst[0][0], worst[0][1], worst[1][0] / max(1, worst[1][1])))

_REPCON_SCANNED = [False]

def _repcon_setup(models, image_pil, clean01, subj, obj, rt, rg, family=None):
    """IDEA-C / METHOD-1. For each model: close any stale tap, then (if enabled
    AND this relation family is in REPCON_FAMILIES) install a LayerTap and build
    the object-word spatial-ID direction. METHOD-1 = relation-geometry-adaptive:
    steer only the families where the spatial-ID helped (default: lateral), and
    leave depth/vertical/containment to the base attack (where steering nulled)."""
    import repcon
    fam_ok = True
    fams = getattr(CFG, "REPCON_FAMILIES", None)
    if fams is not None and family is not None:
        fam_ok = family in fams
    for m in models:
        # tear down a tap left over from the previous example
        old = m.pop("_reptap", None)
        if old is not None:
            old.close()
        m["_repdir"] = None
        m["_rep_last"] = None
        rc = repcon.RepConCfg(
            enable=(CFG.REPCON_ENABLE and fam_ok), layer=CFG.REPCON_LAYER,
            weight=CFG.REPCON_WEIGHT, gamma=CFG.REPCON_GAMMA,
            margin_frac=CFG.REPCON_MARGIN_FRAC,
            dir_target=getattr(CFG, "REPCON_DIR_TARGET", "object"),
            subj_word=subj, obj_word=obj,
            debug=getattr(CFG, "REPCON_DEBUG", True),
        )
        m["repcon_cfg"] = rc
        if CFG.REPCON_ENABLE and not fam_ok and getattr(CFG, "REPCON_DEBUG", True):
            print("    [repcon] METHOD-1: family=%r not in REPCON_FAMILIES=%s "
                  "-> steering OFF (base attack)" % (family, fams))
        if not rc.enable:
            continue
        # helpers repcon.build_direction reads off m (idempotent)
        m["encode_probe"]    = encode_probe
        m["to_pixel_values"] = to_pixel_values
        m["clean01"]         = clean01
        m["_reptap"] = repcon.LayerTap(m["model"], rc.layer)
        # a few TRAIN templates; answers differ ONLY in the relation word.
        n = max(1, int(getattr(CFG, "REPCON_N_DIR_PROBES", 4)))
        idxs = list(POOL["open_train"])[:n] or [0]
        qs = [OPEN_TEMPLATES[i].format(s=subj, o=obj) for i in idxs]
        a_tgt  = f"The {subj} is {rg} the {obj}."
        a_true = f"The {subj} is {rt} the {obj}."
        pairs_target = [(q, a_tgt,  rg) for q in qs]   # relation_phrase = rg
        pairs_true   = [(q, a_true, rt) for q in qs]   # relation_phrase = rt
        if getattr(CFG, "REPCON_SCAN", False) and not _REPCON_SCANNED[0]:
            print("    [repcon-scan] clean-image layer scan (subj=%s obj=%s):" % (subj, obj))
            repcon.scan_layers(m, image_pil, pairs_target, pairs_true,
                               CFG.REPCON_SCAN_LAYERS, rc)
            _REPCON_SCANNED[0] = True
        m["_repdir"] = repcon.build_direction(m, image_pil, pairs_target,
                                              pairs_true, rc, m["_reptap"])
        if not m["_repdir"].ok:
            print("    [repcon] direction FAILED (relation span not found) -> INERT")
        else:
            print("    [repcon] ON layer=%d/%d sep=%.3f" % (
                m["_reptap"].layer_idx, m["_reptap"].n_layers, m["_repdir"].sep))


def craft_phase():
    print("\n########## CRAFT + EVAL (white-box v17) ##########")
    # v18.8 (3): the step pool is shared ACROSS examples, which is the whole
    # mechanism, so it is reset once per RUN and never per example. Without this
    # a resumed run inherits a stale balance and `cost_neutral` -- the gate that
    # decides whether a budget arm's delta is readable at all -- becomes
    # meaningless.
    budget_reset()
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    art_dir = os.path.join(CFG.OUTPUT_DIR, "artifacts"); os.makedirs(art_dir, exist_ok=True)
    restore_from_drive()
    ck = _ckpt_dir()

    done_cache, skip_cache = {}, set()
    n_stale = 0
    if CFG.RESUME:
        for p in sorted(glob.glob(os.path.join(ck, "ex_*.done.json"))):
            try:
                e = _read_json(p)
                for kpng in ("clean_png", "adv_png"):
                    pth = e.get(kpng, "")
                    if pth and not os.path.exists(pth):
                        alt = os.path.join(art_dir, os.path.basename(pth))
                        if os.path.exists(alt): e[kpng] = alt
                # v18: a cached result is only reusable if it was produced under
                # the SAME eval protocol. Mixing a v17 (train-seen) SBC into a
                # v18 (held-out) table would silently inflate the mean.
                _e_arm = str(e.get("baseline_arm", "full"))
                _cur_arm = str(getattr(CFG, "BASELINE_ARM", "full"))
                same_protocol = (bool(e.get("eval_heldout", False)) == bool(CFG.PARAPHRASE_POOL)
                                 and e.get("pool_split_seed") == CFG.POOL_SPLIT_SEED
                                 and (_e_arm == _cur_arm)
                                 and (bool(e.get("pool_reclaim_unused", False))
                                      == bool(CFG.POOL_RECLAIM_UNUSED)))
                if not same_protocol:
                    n_stale += 1
                    continue
                if os.path.exists(e.get("clean_png", "")) and os.path.exists(e.get("adv_png", "")):
                    done_cache[e["key"]] = e
            except Exception as ex:
                print(f"[resume] unreadable {os.path.basename(p)}: {ex!r}")
        for p in glob.glob(os.path.join(ck, "ex_*.skip.json")):
            try: skip_cache.add(_read_json(p)["key"])
            except Exception: pass
        print(f"[resume] {len(done_cache)} done, {len(skip_cache)} skipped.")
        if n_stale:
            print(f"[resume] IGNORED {n_stale} cached result(s) from a different "
                  f"eval protocol (pre-v18 or a different POOL_SPLIT_SEED). They "
                  f"will be recrafted; the old numbers are not comparable.")

    records = load_benchmark_records()
    state = {"models": None, "primary": None}
    def ensure_models():
        if state["models"] is None:
            state["models"] = load_models(); state["primary"] = state["models"][0]
        return state

    from collections import defaultdict
    manifest, results = [], []
    fam_counts = defaultdict(int)
    gate_stats = defaultdict(lambda: {"passed": 0, "failed": 0, "used": 0, "quota_skipped": 0})
    min_per_fam = dict(getattr(CFG, "MIN_PER_FAMILY", {}) or {})
    max_per_fam = dict(getattr(CFG, "MAX_PER_FAMILY", {}) or {})

    def _unmet_rare_quota():
        """How many slots must stay reserved for families still below MIN_PER_FAMILY."""
        return sum(max(0, int(min_q) - fam_counts[f]) for f, min_q in min_per_fam.items())

    used, scanned = 0, 0
    for rec in records:
        if used >= CFG.NUM_EXAMPLES and _unmet_rare_quota() == 0:
            break
        scanned += 1
        img_field, subj, obj, rel_true, label = parse_vsr_record(rec)
        if not (img_field is not None and subj and obj and rel_true): continue
        if CFG.ONLY_TRUE and int(label) != 1: continue
        rel_target = OPPOSITE.get(str(rel_true).lower().strip())
        if rel_target is None: continue
        fam = relation_family(rel_true)
        _of = getattr(CFG, "ONLY_FAMILIES", None)
        if _of and fam not in _of: continue

        is_supp = bool(rec.get("_is_supplement", False))
        src_split = str(rec.get("_source_split", CFG.SPLIT))
        bench_name = str(rec.get("_benchmark", getattr(CFG, "BENCHMARK", "vsr")))
        primary_model_id = CFG.MODEL_IDS[0] if CFG.MODEL_IDS else "unknown"
        gkey = (primary_model_id, bench_name, src_split, fam)

        # In train_supplement: first prioritize any family still below MIN_PER_FAMILY;
        # once all rare families reach MIN_PER_FAMILY (>=20), allow train_supplement
        # to top up the remaining slots to reach CFG.NUM_EXAMPLES (200), up to MAX_PER_FAMILY.
        if is_supp and _unmet_rare_quota() > 0 and fam_counts[fam] >= int(min_per_fam.get(fam, 0)):
            continue

        fam_min = int(min_per_fam.get(fam, 0))
        fam_max = int(max_per_fam.get(fam, CFG.NUM_EXAMPLES))
        remaining_slots = max(0, int(CFG.NUM_EXAMPLES) - used)
        if fam_counts[fam] >= fam_max or (
                fam_counts[fam] >= fam_min and remaining_slots <= _unmet_rare_quota()):
            gate_stats[gkey]["quota_skipped"] += 1
            continue

        image_pil_full, img_name = get_pil_and_name(img_field)
        if image_pil_full is None:
            print(f"[skip] cannot resolve image: {img_name}"); continue

        key = example_key(img_name, subj, obj, rel_true)
        if key in done_cache:
            e = dict(done_cache[key]); e["idx"] = used
            e.setdefault("benchmark", bench_name)
            e.setdefault("source_split", src_split)
            expected_eval_probes = build_eval_probes(subj, obj, rel_true, rel_target)
            _has_old_mc = any(
                "(A)" in str(r.get("question", ""))
                for r in (e.get("probe_log") or [])
                if r.get("probe_key") == "mc_eval"
            )
            if (getattr(CFG, "REEVAL_CACHED_IMAGES", True)
                    and (int(e.get("n_eval_probes", 7)) != len(expected_eval_probes) or _has_old_mc)
                    and os.path.exists(e.get("clean_png", ""))
                    and os.path.exists(e.get("adv_png", ""))):
                st = ensure_models(); primary = st["primary"]
                assert_no_leakage(subj, obj, rel_true, rel_target)
                c_pil = Image.open(e["clean_png"]).convert("RGB")
                a_pil = Image.open(e["adv_png"]).convert("RGB")
                clean01_re = pil_to_img01(c_pil)
                adv01_re   = pil_to_img01(a_pil)
                # Seed preserve_pool from cached preserve_labels so we don't even need to re-decode clean captions
                if e.get("preserve_labels"):
                    primary["preserve_pool"] = [
                        {"q": q, "answer": ans}
                        for q, ans in zip(PRESERVATION_QUESTIONS, e["preserve_labels"])
                    ]
                else:
                    primary["preserve_pool"] = []
                ev_re = evaluate_full(primary, c_pil, clean01_re, adv01_re,
                                      expected_eval_probes, PRESERVATION_QUESTIONS,
                                      rel_true, rel_target, subj=subj, obj=obj)
                sbc_rand_re = e.get("SBC_random")
                if CFG.RUN_RANDOM_CONTROL:
                    set_example_seed(key)
                    adv_rand_re = random_perturbation(clean01_re, float(e.get("eps", CFG.EPS)))
                    rc_re = _sbc_and_coherence(primary, c_pil, adv_rand_re, expected_eval_probes,
                                               rel_true, rel_target, subj=subj, obj=obj)
                    sbc_rand_re = round(rc_re["sbc"], 3)
                e.update({
                    "SBC": ev_re["sbc"],
                    "coherence": ev_re["coherence"],
                    "SBC_nondegen": round(ev_re["sbc_nondegen"], 4),
                    "coherence_nondegen": ev_re["coherence_nondegen"],
                    "n_degenerate": ev_re["n_degenerate"],
                    "SBC_7": round(ev_re["sbc_7"], 4),
                    "coherence_7": ev_re["coherence_7"],
                    "sbc_open": ev_re["sbc_open"],
                    "sbc_yes": ev_re["sbc_yes"],
                    "sbc_no": ev_re["sbc_no"],
                    "preserve_f1": round(ev_re["preserve_f1"], 3),
                    "preserve_strict": ev_re["preserve_strict"],
                    "lpips": compute_perceptual_dist(clean01_re, adv01_re),
                    "SBC_random": sbc_rand_re,
                    "SBC_clean": ev_re.get("SBC_clean"),
                    "SBC_lift": ev_re.get("SBC_lift"),
                    "coherence_clean": ev_re.get("coherence_clean"),
                    "sbc_open_clean": ev_re.get("sbc_open_clean"),
                    "sbc_yes_clean": ev_re.get("sbc_yes_clean"),
                    "sbc_no_clean": ev_re.get("sbc_no_clean"),
                    "sbc_yes_lift": ev_re.get("sbc_yes_lift"),
                    "sbc_no_lift": ev_re.get("sbc_no_lift"),
                    "per_frame": ev_re.get("per_frame"),
                    "per_frame_clean": ev_re.get("per_frame_clean"),
                    "clean_probe_log": ev_re.get("clean_probe_log"),
                    "n_eval_probes": len(expected_eval_probes),
                    "probe_log": ev_re["rows"],
                })
                _atomic_write_json(_ex_done_path(key), e)
                print(f"[reeval-15] ex{used:03d} ({img_name}, fam={fam}) -> "
                      f"SBC_15={e['SBC']:.3f} coh_15={e['coherence']:.0f} "
                      f"(legacy7: SBC_7={e['SBC_7']:.3f} coh_7={e['coherence_7']:.0f})")
            else:
                print(f"[resume] ex{used:03d} reused ({img_name}, fam={fam}[{fam_counts[fam]+1}])")
            manifest.append({k: v for k, v in e.items() if k != "probe_log"})
            results.append(e)
            fam_counts[fam] += 1
            gate_stats[gkey]["passed"] += 1
            gate_stats[gkey]["used"] += 1
            used += 1; continue
        if key in skip_cache:
            gate_stats[gkey]["failed"] += 1
            continue

        st = ensure_models(); models, primary = st["models"], st["primary"]
        n_steps = CFG.STEPS_BY_FAMILY.get(fam, CFG.STEPS) if hasattr(CFG, "STEPS_BY_FAMILY") else CFG.STEPS
        _base_steps = n_steps
        n_steps = compensated_steps(n_steps)
        n_eot   = CFG.EOT_SAMPLES_BY_FAMILY.get(fam, CFG.EOT_SAMPLES) if hasattr(CFG, "EOT_SAMPLES_BY_FAMILY") else CFG.EOT_SAMPLES
        print(f"\n=== ex {used} [{bench_name}/{src_split}]: '{subj}'/'{obj}' | "
              f"true='{rel_true}' -> target='{rel_target}' | "
              f"family={fam} (count={fam_counts[fam] + 1}, min={fam_min}) | {img_name}")
        print(f"    [opt4] steps={n_steps} (base {_base_steps} x {n_train_variants()}/3 "
              f"variants) eot_samples={n_eot}  antithetic={CFG.EOT_ANTITHETIC}")
        image_pil = image_pil_full.resize((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE))
        clean01 = pil_to_img01(image_pil)

        ok, dbg = clean_is_correct(primary, image_pil, clean01, subj, obj, rel_true, rel_target)
        if not ok:
            gate_stats[gkey]["failed"] += 1
            print(f"[skip] clean gate failed (true->'{dbg['true_q']}', target->'{dbg['target_q']}')")
            _atomic_write_json(_ex_skip_path(key),
                               {"key": key, "benchmark": bench_name, "source_split": src_split,
                                "family": fam, "image": img_name,
                                "reason": "clean_gate_failed", "dbg": dbg})
            continue
        gate_stats[gkey]["passed"] += 1
        gate_stats[gkey]["used"] += 1
        fam_counts[fam] += 1

        for m in models:
            build_model_probes(m, image_pil, subj, obj, rel_true, rel_target)
            # v18.3 (A): freeze the CLEAN answers as preservation labels. Must
            # happen here, before any perturbation exists, and once per example
            # -- rebuilding them per sweep run would let the labels drift.
            build_preserve_rows(m, image_pil, clean01)
        eval_probes = build_eval_probes(subj, obj, rel_true, rel_target)
        assert_no_leakage(subj, obj, rel_true, rel_target)

        # ---- IDEA-C: representation-space belief consistency setup ----------
        # Build the target-vs-true relation direction ONCE per example on the
        # clean image, from TRAIN templates only (never eval -> no leakage).
        _repcon_setup(models, image_pil, clean01, subj, obj, rel_true, rel_target, fam)
        # LOSS_MODE: counterfactual-teacher (mirror) loss -- per-example setup.
        _cf_info = _cf_setup(models, image_pil, clean01, subj, obj, rel_true, rel_target, fam)

        if CFG.DEBUG_ATTACK:
            _debug_idea1(subj, obj, rel_true, rel_target)
            _debug_options(primary, subj, obj, rel_true, rel_target)

        # ---- OPT-C sweep: attack once per FREERUN_WEIGHT value, keep the best.
        # Same seed across values => fair comparison. Since 0.0 == exact v14, the
        # kept result is never worse than the v14 baseline.
        # v16 OPT-H: the sweep dimension is now configurable.
        #   "restart" -> N seeds at the single best setting  (v16 default)
        #   "freerun" -> v15 behaviour, sweep FREERUN_SWEEP
        #   "off"     -> one run
        mode = getattr(CFG, "SWEEP_MODE", "freerun")
        if mode == "restart":
            sweep_specs = [{"fw": CFG.FREERUN_WEIGHT, "restart": i}
                           for i in range(max(1, CFG.RESTART_SWEEP))]
        elif mode == "freerun":
            vals = list(CFG.FREERUN_SWEEP) if CFG.FREERUN_SWEEP else [CFG.FREERUN_WEIGHT]
            sweep_specs = [{"fw": v, "restart": 0} for v in vals]
        elif mode == "hybrid":
            vals = list(CFG.FREERUN_SWEEP) if CFG.FREERUN_SWEEP else [CFG.FREERUN_WEIGHT]
            sweep_specs = [{"fw": v, "restart": i}
                           for v in vals for i in range(max(1, CFG.RESTART_SWEEP))]
        else:
            sweep_specs = [{"fw": CFG.FREERUN_WEIGHT, "restart": 0}]

        # ---- v18.2 CURVE MODE. Replaces the sweep with a one-axis parameter
        # scan whose points are ALL kept. This is not a search for the best
        # setting: picking the best eps per example would be selection on the
        # test metric, which is what the leakage fix was meant to stop.
        curve_axis = None
        if CFG.CURVE_MODE == "eps":
            curve_axis = "eps"
            sweep_specs = [{"fw": CFG.FREERUN_WEIGHT, "restart": 0, "eps": float(v)}
                           for v in CFG.EPS_CURVE_VALUES]
        elif CFG.CURVE_MODE == "mask":
            curve_axis = "mask_keep"
            sweep_specs = [{"fw": CFG.FREERUN_WEIGHT, "restart": 0, "mask_keep": float(v)}
                           for v in CFG.MASK_CURVE_VALUES]
        elif CFG.CURVE_MODE == "preserve_c":
            # v18.3: the Pareto curve drawn on the CONSTRAINT rather than on
            # attack capacity. Each point is a controlled trade-off ("how much
            # caption drift will you buy the lie with?"), whereas eps and mask
            # points differ in how much the attack is ALLOWED to do at all.
            curve_axis = "preserve_c"
            sweep_specs = [{"fw": CFG.FREERUN_WEIGHT, "restart": 0,
                            "preserve_c": float(v)}
                           for v in CFG.PRESERVE_CURVE_VALUES]
        elif CFG.CURVE_MODE == "probe_c":
            # v18.7 IDEA-A: the coherence curve drawn on the PER-PROBE
            # constraint. Loose C == cheap probes == the v18.6 sum objective;
            # tight C == every probe must actually be bought. The certificate
            # count at each point says how many probes are simply not for sale
            # at this eps.
            curve_axis = "probe_c"
            sweep_specs = [{"fw": CFG.FREERUN_WEIGHT, "restart": 0,
                            "probe_c": float(v)}
                           for v in CFG.PROBE_C_CURVE_VALUES]

        _orig_fw = CFG.FREERUN_WEIGHT
        _orig_pc = CFG.PRESERVE_C
        _orig_probe_c = CFG.PROBE_C          # v18.7 IDEA-A curve axis
        best = None            # (score_tuple, spec, adv01, hist, ev, linf)
        sweep_rows = []
        runs = []              # every point, in spec order (curve mode uses this)
        _mask_cache = {}
        for spec in sweep_specs:
            fw, rs = spec["fw"], spec["restart"]
            eps_i = float(spec.get("eps", CFG.EPS))
            mk_i = float(spec.get("mask_keep", CFG.MASK_KEEP_FRAC))
            pc_i = float(spec.get("preserve_c", CFG.PRESERVE_C))
            prc_i = float(spec.get("probe_c", _orig_probe_c))   # v18.7 IDEA-A
            CFG.FREERUN_WEIGHT = fw
            CFG.PRESERVE_C = pc_i
            CFG.PROBE_C = prc_i
            for mm in models:
                mm["probe_ema"] = {}                    # fresh OPT-B EMA per run
                # v18.5: same contract as probe_ema. A sweep point that
                # inherited the previous point's fail-fraction would start with
                # a tail budget fitted to a different perturbation.
                mm["_op_fail_frac"] = None
                mm["_cal"] = {"a": float(CFG.CVAR_CAL_A0),
                              "b": float(CFG.CVAR_CAL_B0), "n": 0}
                mm["_fam_fail"] = {}          # v18.6 FMR state is per-RUN too
                mm["_fam_mult_last"] = {}
                mm["onpolicy_pool"] = []                # IDEA-2: fresh FIFO per run
                mm["_op_stats"] = {"triggers": 0, "rollouts": 0, "minted": 0,
                                   "boosted": 0, "open_fail": 0, "yn_fail": 0,
                                   "mint_skipped": 0, "mint_gated": 0,
                                   "gate_reasons": {}}
                # v18.3: the dual variable is per-RUN state, like probe_ema.
                # Carrying it across sweep points would leak the previous
                # point's constraint pressure into the next one and quietly
                # destroy the curve's independence. The frozen LABELS
                # (preserve_pool) are per-EXAMPLE and are NOT reset here.
                mm["preserve_lambda"] = float(CFG.PRESERVE_WEIGHT)
                mm["_preserve_last"] = None
                mm["_lambda_history"] = []
                # v18.7 IDEA-A: the per-probe duals are per-RUN state, exactly
                # like probe_ema and preserve_lambda. Carrying them across sweep
                # points would leak one point's constraint pressure into the
                # next and destroy the independence of the PROBE_C curve.
                init_probe_lagrangian(mm)
            # set_example_seed also resets the CRN stream (IDEA-B), so every
            # sweep point at the same salt sees identical noise across arms.
            set_example_seed(key, salt=rs)              # salt=0 == v15 seeding

            # Saliency mask is derived from the CLEAN image, so it is identical
            # across specs that share keep_frac -> build once, reuse.
            mask_i = None
            if CFG.MASK_ENABLE and mk_i < 1.0:
                if mk_i not in _mask_cache:
                    _mask_cache[mk_i] = build_spatial_mask(
                        models, clean01, CFG.PROMPT_EOT, n_eot, keep_frac=mk_i)
                mask_i = _mask_cache[mk_i]
            elif getattr(CFG, "NOVEL_SOFT_ENVELOPE", False) and str(getattr(CFG, "BASELINE_ARM", "full")) not in ("B0", "B1", "B2", "B3", "B4", "B5"):
                if "soft_env" not in _mask_cache:
                    _mask_cache["soft_env"] = build_soft_entity_envelope(
                        models, clean01, CFG.PROMPT_EOT, n_eot)
                mask_i = _mask_cache["soft_env"]
            mask_frac_i = None if mask_i is None else round(float(mask_i.mean()), 4)

            _tg = f"cw-ste[fr={fw},r={rs}"
            if curve_axis == "eps":       _tg += f",eps={eps_i*255:.0f}/255"
            if curve_axis == "mask_keep": _tg += f",keep={mk_i:.2f}"
            _t_ex0 = time.time()
            _b_arm = str(getattr(CFG, "BASELINE_ARM", "full"))
            if _b_arm == "B0":
                set_example_seed(key, salt=rs)
                adv_i = random_perturbation(clean01, eps_i)
                hist_i = [0.0]
            elif _b_arm == "B4":
                adv_i, hist_i = cropa_attack_b4(
                    models, clean01, eps=eps_i, n_iter=n_steps,
                    prompt_eot=CFG.PROMPT_EOT, eot_samples=n_eot, tag=f"B4-CroPA[r={rs}]")
            elif _b_arm == "B5":
                adv_i, hist_i = feature_match_attack_b5(
                    models, clean01, eps=eps_i, n_iter=n_steps,
                    fam=fam, tag=f"B5-feat[r={rs}]")
            else:
                adv_i, hist_i = apgd_attack(models, clean01, eps_i, n_steps,
                                            CFG.PROMPT_EOT, random_init=CFG.RANDOM_INIT,
                                            tag=_tg + "]", eot_samples=n_eot,
                                            mask=mask_i)
            _wall_time_i = round(time.time() - _t_ex0, 2)
            ev_i = evaluate_full(primary, image_pil, clean01, adv_i, eval_probes,
                                 PRESERVATION_QUESTIONS, rel_true, rel_target,
                                 subj=subj, obj=obj)
            linf_i = (adv_i - clean01).abs().max().item()
            op_i = dict(models[0].get("_op_stats", {}))
            sweep_rows.append({"freerun_weight": fw, "restart": rs,
                               "eps": round(eps_i, 6),
                               "eps_255": round(eps_i * 255, 2),
                               "mask_keep": mk_i if CFG.MASK_ENABLE else None,
                               "mask_frac_actual": mask_frac_i,
                               "linf": round(linf_i, 5),
                               "coherence_nondegen": ev_i["coherence_nondegen"],
                               "preserve_strict": round(ev_i["preserve_strict"], 3),
                               "SBC": round(ev_i["sbc"], 3),
                               "SBC_nondegen": round(ev_i["sbc_nondegen"], 3),
                               "n_degenerate": ev_i["n_degenerate"],
                               "coherence": ev_i["coherence"], "sbc_open": ev_i["sbc_open"],
                               "sbc_yes": ev_i["sbc_yes"], "sbc_no": ev_i["sbc_no"],
                               "preserve_f1": round(ev_i["preserve_f1"], 3),
                               "onpolicy": op_i,
                               "preserve_c": pc_i if CFG.PRESERVE_LOSS else None,
                               "preserve_ce_final": (models[0].get("_preserve_last")
                                                     if CFG.PRESERVE_LOSS else None),
                               "preserve_lambda_final": (models[0].get("preserve_lambda")
                                                         if CFG.PRESERVE_LOSS else None),
                               # --- v18.7 IDEA-A: the curve point needs the
                               # certificate count, or "coherence fell at
                               # tight C" is indistinguishable from "the
                               # constraint was never satisfiable".
                               "probe_c": prc_i if CFG.PROBE_LAGRANGIAN else None,
                               "probe_cert": ((lambda c: {k: c[k] for k in
                                               ("n_probes", "n_satisfied",
                                                "n_certified_infeasible",
                                                "lambda_mean", "lambda_sd",
                                                "lambda_cv")})(
                                                   probe_certificates(models[0]))
                                              if CFG.PROBE_LAGRANGIAN else None),
                               "crn_draws": int(_CRN.get("draws", 0)),
                               "final_loss": hist_i[-1] if hist_i else None})
            runs.append((spec, adv_i, hist_i, ev_i, linf_i, eps_i, mk_i, mask_frac_i))
            if curve_axis:
                print(f"    [curve] {curve_axis}="
                      + (f"{eps_i*255:.0f}/255" if curve_axis == "eps"
                         else f"{prc_i:.2f}" if curve_axis == "probe_c"
                         else f"{pc_i:.2f}" if curve_axis == "preserve_c"
                         else f"{mk_i:.2f}")
                      + f"  SBC={ev_i['sbc']:.3f} coh={ev_i['coherence']:.0f} "
                        f"open={ev_i['sbc_open']} "
                        f"pf1={ev_i['preserve_f1']:.3f} "
                        f"pstrict={ev_i['preserve_strict']:.2f} Linf={linf_i:.4f}")
            print(f"    [sweep] fr_w={fw:<5} r={rs} SBC={ev_i['sbc']:.3f} "
                  f"(nondegen={ev_i['sbc_nondegen']:.3f}, degen={ev_i['n_degenerate']}) "
                  f"coh={ev_i['coherence']:.0f} "
                  f"open={ev_i['sbc_open']} yes={ev_i['sbc_yes']} no={ev_i['sbc_no']} "
                  f"preserve_f1={ev_i['preserve_f1']:.2f} Linf={linf_i:.4f}")
            if CFG.ONPOLICY and op_i:
                print(f"    [onpol] triggers={op_i.get('triggers',0)} "
                      f"rollouts={op_i.get('rollouts',0)} "
                      f"open_fail={op_i.get('open_fail',0)} minted={op_i.get('minted',0)} "
                      f"gated={op_i.get('mint_gated',0)}{op_i.get('gate_reasons') or ''} "
                      f"skipped={op_i.get('mint_skipped',0)} "
                      f"yn_fail={op_i.get('yn_fail',0)} boosted={op_i.get('boosted',0)}")
            # rank: higher SBC, then coherence, then preservation
            st = (round(ev_i["sbc"], 6), round(ev_i["coherence"], 6), round(ev_i["preserve_f1"], 6))
            if best is None or st > best[0]:
                best = (st, spec, adv_i, hist_i, ev_i, linf_i)
            # early exit: a perfect example cannot be improved by more restarts.
            # NEVER in curve mode -- a curve with missing points is not a curve.
            if (not curve_axis) and getattr(CFG, "SWEEP_EARLY_EXIT", True) \
                    and st[0] >= 1.0 and st[1] >= 1.0:
                print(f"    [sweep] perfect at r={rs}; skipping remaining restarts.")
                break
        CFG.FREERUN_WEIGHT = _orig_fw
        CFG.PRESERVE_C = _orig_pc
        CFG.PROBE_C = _orig_probe_c

        # In curve mode the headline entry comes from a FIXED reference point,
        # not from whichever point scored highest.
        if curve_axis and runs:
            if curve_axis == "eps":
                ref = min(runs, key=lambda r: abs(r[5] - float(CFG.CURVE_REFERENCE)))
            elif curve_axis == "preserve_c":
                # Loosest constraint = closest to "no preservation term at all",
                # which is what keeps this row comparable to every pre-v18.3 run.
                ref = max(runs, key=lambda r: float(r[0].get("preserve_c", 0.0)))
            elif curve_axis == "probe_c":
                # Loosest constraint = closest to the v18.6 weighted-sum
                # objective, which is what keeps this row comparable to every
                # pre-v18.7 run. NOT the best-scoring point.
                ref = max(runs, key=lambda r: float(r[0].get("probe_c", 0.0)))
            else:
                ref = max(runs, key=lambda r: r[6])
            best = ((round(ref[3]["sbc"], 6), round(ref[3]["coherence"], 6),
                     round(ref[3]["preserve_f1"], 6)),
                    ref[0], ref[1], ref[2], ref[3], ref[4])
            print(f"    [curve] headline point = "
                  + (f"eps={ref[5]*255:.0f}/255" if curve_axis == "eps"
                     else f"keep={ref[6]:.2f}")
                  + "  (fixed reference, NOT the best point)")

        _, spec_used, adv01, hist, ev, linf = best
        eps_used = float(spec_used.get("eps", CFG.EPS))
        fw_used, restart_used = spec_used["fw"], spec_used["restart"]

        # Decoupling measurement (diagnostic): does the OUTPUT flip while the
        # relation-word logit gap still favours TRUE? (the "bed" phenomenon)
        _open_diag = None
        if getattr(CFG, "OPEN_DIAG", False):
            try:
                _open_diag = open_diag(models[0], image_pil, clean01, adv01, eval_probes,
                                       subj, obj, rel_true, rel_target)
            except Exception as _e:
                print("    [odiag] failed: %r" % (_e,))
        # v18.10 + DICD MBPD diagnostic (all 5 families: mirror on lateral, dual on others).
        _mbpd = None
        if getattr(CFG, "MBPD_DIAG", False):
            try:
                _mbpd = mbpd_diag(models[0], image_pil, clean01, adv01,
                                  subj, obj, rel_true, rel_target)
            except Exception as _e:
                print("    [mbpd] failed: %r" % (_e,))
        _decouple_cert = None
        if getattr(CFG, "REPCON_DECOUPLE", False):
            try:
                import repcon
                m0 = models[0]
                m0.setdefault("encode_probe", encode_probe)
                m0.setdefault("to_pixel_values", to_pixel_values)
                m0.setdefault("generate", generate)
                m0.setdefault("score_lie", score_lie)
                _decouple_cert = repcon.measure_decoupling(
                    m0, image_pil, clean01, adv01, eval_probes,
                    rel_true, rel_target, subj, obj)
            except Exception as _e:
                print("    [decouple] failed: %r" % (_e,))

        # IDEA-3: causal patching probe (diagnostic, no effect on the attack).
        if getattr(CFG, "REPCON_PATCH_PROBE", False):
            try:
                import repcon
                m0 = models[0]
                m0.setdefault("encode_probe", encode_probe)
                m0.setdefault("to_pixel_values", to_pixel_values)
                m0.setdefault("processor", m0.get("processor"))
                print("    [patch] === IDEA-3 causal probe (subj=%s obj=%s true=%s target=%s) ==="
                      % (subj, obj, rel_true, rel_target))
                repcon.patch_probe(m0, image_pil, clean01, adv01, subj, obj,
                                   rel_true, rel_target,
                                   layers=CFG.REPCON_PATCH_LAYERS)
            except Exception as _e:
                print("    [patch] probe failed: %r" % (_e,))

        if len(sweep_specs) > 1:
            print(f"    [sweep] -> BEST fr_w={fw_used} r={restart_used}  SBC={ev['sbc']:.3f} "
                  f"coh={ev['coherence']:.0f} preserve_f1={ev['preserve_f1']:.2f}")
        print(f"    [best] SBC={ev['sbc']:.2f} SBC_nondegen={ev['sbc_nondegen']:.2f} "
              f"degen_probes={ev['n_degenerate']} coherence={ev['coherence']:.0f} "
              f"open={ev['sbc_open']} yes={ev['sbc_yes']} no={ev['sbc_no']} "
              f"preserve_f1={ev['preserve_f1']:.2f} Linf={linf:.4f} "
              f"fr_w={fw_used} restart={restart_used}")

        sbc_random = None
        if CFG.RUN_RANDOM_CONTROL:
            set_example_seed(key)
            adv_rand = random_perturbation(clean01, eps_used)
            rc = _sbc_and_coherence(primary, image_pil, adv_rand, eval_probes,
                                    rel_true, rel_target, subj=subj, obj=obj)
            sbc_random = round(rc["sbc"], 3)
            print(f"    [random] SBC_random={sbc_random:.2f}")

        clean_png = os.path.join(art_dir, f"clean_{used:03d}_{key}.png")
        adv_png   = os.path.join(art_dir, f"adv_{used:03d}_{key}.png")
        _atomic_save_pil(img01_to_pil(clean01), clean_png)
        _atomic_save_pil(img01_to_pil(adv01),   adv_png)

        entry = {"key": key, "idx": used, "benchmark": bench_name, "source_split": src_split,
                 "image": img_name, "subj": subj, "obj": obj,
                 "rel_true": rel_true, "rel_target": rel_target, "family": fam,
                 "clean_png": clean_png, "adv_png": adv_png,
                 "SBC": ev["sbc"], "coherence": ev["coherence"],
                 "SBC_nondegen": round(ev["sbc_nondegen"], 4),
                 "coherence_nondegen": ev["coherence_nondegen"],
                 "n_degenerate": ev["n_degenerate"],
                 "SBC_7": round(ev.get("sbc_7", ev["sbc"]), 4),
                 "coherence_7": ev.get("coherence_7", ev["coherence"]),
                 "sbc_open": ev["sbc_open"], "sbc_yes": ev["sbc_yes"], "sbc_no": ev["sbc_no"],
                 "preserve_f1": round(ev["preserve_f1"], 3),
                 "preserve_strict": ev["preserve_strict"], "linf": linf,
                 "lpips": compute_perceptual_dist(clean01, adv01),
                 "wall_time_s": _wall_time_i,
                 "baseline_arm": str(getattr(CFG, "BASELINE_ARM", "full")),
                 "seed": int(CFG.SEED),
                 "SBC_random": sbc_random,
                 # --- v18.8 (2): THE CLEAN-IMAGE CONTROL.
                 # SBC_lift is the quantity attributable to the ATTACK.
                 # SBC alone is not: a model with an assent prior scores ~1.0 on
                 # the yes-probes of the clean image, and those are 2 of the 7.
                 # Report SBC_lift beside SBC in every table.
                 "SBC_clean": ev.get("SBC_clean"),
                 "SBC_lift": ev.get("SBC_lift"),
                 "coherence_clean": ev.get("coherence_clean"),
                 "sbc_open_clean": ev.get("sbc_open_clean"),
                 "sbc_yes_clean": ev.get("sbc_yes_clean"),
                 "sbc_no_clean": ev.get("sbc_no_clean"),
                 "sbc_yes_lift": ev.get("sbc_yes_lift"),
                 "sbc_no_lift": ev.get("sbc_no_lift"),
                 "per_frame": ev.get("per_frame"),
                 "per_frame_clean": ev.get("per_frame_clean"),
                 "clean_probe_log": ev.get("clean_probe_log"),
                 "final_loss": hist[-1] if hist else None, "steps": n_steps,
                 "eps": eps_used, "eps_255": round(eps_used * 255, 2),
                 "freerun_weight_used": fw_used, "restart_used": restart_used,
                 "freerun_sweep": sweep_rows, "sweep_mode": mode,
                 "n_sweep_runs": len(sweep_rows),
                 "preset": CFG.PRESET, "base_steps": _base_steps,
                 "n_train_variants": n_train_variants(),
                 "compensate_steps": CFG.COMPENSATE_STEPS,
                 "append_frame": CFG.ADD_APPEAR_FRAME,
                 "filler_evasion": CFG.ADD_FILLER_EVASION,
                 "yesno_contrast": CFG.YESNO_CONTRAST,
                 "multi_ref_yesno": CFG.MULTI_REF_YESNO,
                 "method": "apgd-whitebox-v17-preset-" + str(CFG.PRESET),
                 "multi_ref": CFG.MULTI_REF, "forced_choice": CFG.ADD_FORCED_CHOICE,
                 "margin_weight": CFG.MARGIN_WEIGHT, "ce_weight": CFG.CE_WEIGHT,
                 "margin": CFG.MARGIN, "pivot_only": CFG.PIVOT_ONLY,
                 "contrast_weight": CFG.CONTRAST_WEIGHT, "contrast_margin": CFG.CONTRAST_MARGIN,
                 "evasion_weight": CFG.EVASION_WEIGHT, "evasion_margin": CFG.EVASION_MARGIN,
                 "hardprobe_weighting": CFG.HARDPROBE_WEIGHTING,
                 "hardprobe_tau": CFG.HARDPROBE_TAU, "hardprobe_beta": CFG.HARDPROBE_BETA,
                 "freerun_weight": CFG.FREERUN_WEIGHT, "freerun_margin": CFG.FREERUN_MARGIN,
                 "quant_ste": CFG.QUANT_STE, "eot_samples": n_eot,
                 "eot_antithetic": CFG.EOT_ANTITHETIC,
                 "eot_dither_lsb": round(CFG.EOT_DITHER * 255, 3),
                 "max_rows_per_fwd": CFG.MAX_ROWS_PER_FWD, "grad_ckpt": CFG.GRAD_CHECKPOINT,
                 "prompt_eot": CFG.PROMPT_EOT,
                 # --- v18.8. Recorded on EVERY run, including arms where they
                 #     are off, so compare_arms.py can diff two files
                 #     field-by-field before their numbers are compared.
                 "pool_reclaim_unused": CFG.POOL_RECLAIM_UNUSED,
                 "pool_eval_audit": dict(_EVAL_SPLIT_AUDIT),
                 "contrast_margin_no": CFG.CONTRAST_MARGIN_NO,
                 "polarity_invar": CFG.POLARITY_INVAR,
                 "polarity_invar_w": CFG.POLARITY_INVAR_W,
                 # THE INERTNESS TEST for (1c): the variance term needs >= 2
                 # no-probes inside one micro-batch. If fires == 0 the term
                 # never ran and cannot have contributed, whatever SBC says.
                 "polarity_invar_fires": int(models[0].get("_polinv_fires", 0)),
                 "polarity_invar_skips": int(models[0].get("_polinv_skips", 0)),
                 "eval_clean_control": CFG.EVAL_CLEAN_CONTROL,
                 "eval_per_frame": CFG.EVAL_PER_FRAME,
                 "budget_adapt": CFG.BUDGET_ADAPT,
                 "budget_decide_frac": CFG.BUDGET_DECIDE_FRAC,
                 "budget_low": CFG.BUDGET_LOW, "budget_high": CFG.BUDGET_HIGH,
                 "budget_bonus": CFG.BUDGET_BONUS,
                 "budget_max_mult": CFG.BUDGET_MAX_MULT,
                 "budget_report": (budget_report() if CFG.BUDGET_ADAPT else None),
                 # --- v18 ---
                 "paraphrase_pool": CFG.PARAPHRASE_POOL,
                 "dro_cvar": CFG.DRO_CVAR,
                 "cvar_alpha_final": CFG.CVAR_ALPHA_FINAL,
                 "cvar_anneal": CFG.CVAR_ANNEAL,
                 "dro_subset": CFG.DRO_SUBSET,
                 # --- v18.5. Recorded on EVERY run, including the arms where
                 #     they are off, so two result files can always be diffed
                 #     field-by-field before their numbers are compared. The
                 #     v18.4 postmortem lost a week to exactly that check not
                 #     being possible.
                 "cvar_effective": (
                     "uniform" if (not CFG.DRO_CVAR and not CFG.HARDPROBE_WEIGHTING)
                     else "softmax_ema" if not CFG.DRO_CVAR
                     else "uniform(alpha=1)" if (float(CFG.CVAR_ALPHA_FINAL) >= 1.0
                                                 and not CFG.CVAR_ADAPTIVE)
                     else "cvar"),
                 "cvar_adaptive": CFG.CVAR_ADAPTIVE,
                 "cvar_adapt_delta": CFG.CVAR_ADAPT_DELTA,
                 "cvar_alpha_floor": CFG.CVAR_ALPHA_FLOOR,
                 "cvar_calibrate": CFG.CVAR_CALIBRATE,
                 "cvar_typed": CFG.CVAR_TYPED,
                 "cvar_mix": CFG.CVAR_MIX,
                 "cvar_coldstart_inf": CFG.CVAR_COLDSTART_INF,
                 "onpolicy_boost_mode": getattr(CFG, "ONPOLICY_BOOST_MODE", "ema"),
                 "onpolicy_fail_boost": CFG.ONPOLICY_FAIL_BOOST,
                 "cvar_fail_frac_final": (
                     None if models[0].get("_op_fail_frac") is None
                     else round(float(models[0]["_op_fail_frac"]), 4)),
                 "cvar_cal_final": dict(models[0].get("_cal") or {}),
                 # --- v18.6 FMR ---
                 "family_reweight": CFG.FAMILY_REWEIGHT,
                 "family_rw_eps": CFG.FAMILY_RW_EPS,
                 "family_rw_beta": CFG.FAMILY_RW_BETA,
                 "family_rw_min": CFG.FAMILY_RW_MIN,
                 "family_rw_max": CFG.FAMILY_RW_MAX,
                 "family_fail_final": dict(models[0].get("_fam_fail") or {}),
                 "family_mult_final": dict(models[0].get("_fam_mult_last") or {}),
                 "onpolicy": CFG.ONPOLICY,
                 "onpolicy_every": CFG.ONPOLICY_EVERY,
                 "onpolicy_prefix_tokens": CFG.ONPOLICY_PREFIX_TOKENS,
                 "onpolicy_buffer": CFG.ONPOLICY_BUFFER,
                 "n_train_probes": len(models[0].get("train_pool", [])),
                 "n_eval_probes": len(eval_probes),
                 "eval_heldout": bool(CFG.PARAPHRASE_POOL),
                 "pool_split_seed": CFG.POOL_SPLIT_SEED,
                 # --- v18.1 telemetry + scorer provenance ---
                 "onpolicy_stats": dict(models[0].get("_op_stats", {})),
                 "onpolicy_rows_final": len(models[0].get("onpolicy_pool", [])),
                 # --- v18.2 ---
                 "curve_mode": CFG.CURVE_MODE,
                 "curve": (sweep_rows if curve_axis else None),
                 "curve_axis": curve_axis,
                 "mask_enable": CFG.MASK_ENABLE,
                 "mask_mode": CFG.MASK_MODE,
                 "mask_keep_frac": (CFG.MASK_KEEP_FRAC if CFG.MASK_ENABLE else None),
                 "mask_frac_actual": next((r[7] for r in runs
                                           if r[0] is spec_used), None),
                 "mask_blur_sigma": CFG.MASK_BLUR_SIGMA,
                 "mask_dilate": CFG.MASK_DILATE,
                 "onpolicy_mint_gate": CFG.ONPOLICY_MINT_GATE,
                 # --- v18.3 ---
                 "preserve_loss": CFG.PRESERVE_LOSS,
                 "preserve_lagrangian": CFG.PRESERVE_LAGRANGIAN,
                 "preserve_c": (CFG.PRESERVE_C if CFG.PRESERVE_LOSS else None),
                 "preserve_c_used": (float(spec_used.get("preserve_c", CFG.PRESERVE_C))
                                     if CFG.PRESERVE_LOSS else None),
                 "preserve_weight_init": CFG.PRESERVE_WEIGHT,
                 "preserve_lambda_lr": CFG.PRESERVE_LAMBDA_LR,
                 "preserve_lambda_final": (models[0].get("preserve_lambda")
                                           if CFG.PRESERVE_LOSS else None),
                 "preserve_ce_final": (models[0].get("_preserve_last")
                                       if CFG.PRESERVE_LOSS else None),
                 "preserve_labels": [r["answer"] for r in
                                     (models[0].get("preserve_pool") or [])],
                 # --- v18.7 IDEA-A: per-probe Lagrangian ------------------
                 "probe_lagrangian": CFG.PROBE_LAGRANGIAN,
                 "probe_c": (CFG.PROBE_C if CFG.PROBE_LAGRANGIAN else None),
                 "probe_c_used": (float(spec_used.get("probe_c", CFG.PROBE_C))
                                  if CFG.PROBE_LAGRANGIAN else None),
                 "probe_lambda_lr": CFG.PROBE_LAMBDA_LR,
                 "probe_lambda_max": CFG.PROBE_LAMBDA_MAX,
                 "probe_lambda_normalize": CFG.PROBE_LAMBDA_NORMALIZE,
                 # The certificate table. `lambda_cv` is the inertness test:
                 # near zero means the duals never differentiated between
                 # probes and IDEA-A contributed nothing this run, whatever
                 # happened to SBC.
                 "probe_certificates": (probe_certificates(models[0])
                                        if CFG.PROBE_LAGRANGIAN else None),
                 "probe_lambda_trace": (models[0].get("_probe_lam_trace")
                                        if (CFG.PROBE_LAGRANGIAN and
                                            CFG.PROBE_LAG_LOG_TRACE) else None),
                 # --- IDEA-C: representation-space belief consistency ------
                 "repcon_enable": CFG.REPCON_ENABLE,
                 "repcon_layer": (models[0].get("_reptap").layer_idx
                                  if (CFG.REPCON_ENABLE and
                                      models[0].get("_reptap") is not None) else None),
                 "repcon_sep": (float(models[0]["_repdir"].sep)
                                if (CFG.REPCON_ENABLE and
                                    models[0].get("_repdir") is not None and
                                    models[0]["_repdir"].ok) else None),
                 "repcon_cert": (models[0].get("_rep_last")
                                 if CFG.REPCON_ENABLE else None),
                 "decouple_cert": _decouple_cert,
                 "loss_mode": getattr(CFG, "LOSS_MODE", "classic"),
                 "cf_info": _cf_info,
                 "cf_stats": _cf_report(models[0]),
                 "cf_contrast_cfg": ({"alpha": CFG.CF_CONTRAST_ALPHA,
                                      "beta": CFG.CF_CONTRAST_BETA,
                                      "js": CFG.CF_CONTRAST_JS,
                                      "js_ref": CFG.CF_CONTRAST_JS_REF,
                                      "replace_gate": CFG.CF_CONTRAST_REPLACE_GATE,
                                      "open_swing": CFG.CF_CONTRAST_OPEN}
                                     if getattr(CFG, "CF_CONTRAST", False) else None),
                 # v18.8: did the open-dir teacher earn its place on this example?
                 # open_dir_lift = sbc_open - sbc_open_clean (attack-attributable
                 # gain on OPEN rows). Only meaningful when open_dir_on is True.
                 # A negative value means the teacher HURT vs the clean baseline.
                 "open_dir_helped": (
                     None if not (_cf_info or {}).get("open_dir_on")
                     else round(float(ev["sbc_open"]) - float(ev.get("sbc_open_clean") or 0.0), 4)),
                 "open_diag": _open_diag,
                 "mbpd_diag": _mbpd,
                 "mbpd_loss": bool(getattr(CFG, "MBPD_LOSS", False)),
                 # --- v18.7 IDEA-B: common random numbers -----------------
                 "crn": crn_report(key),
                 "deterministic": CFG.DETERMINISTIC,
                 "ce_nonpivot_only": CFG.CE_NONPIVOT_ONLY,
                 "v183_yesno_contrast": CFG.V183_YESNO_CONTRAST,
                 "apgd_cond1_mode": CFG.APGD_COND1_MODE,
                 "steps_scale": CFG.STEPS_SCALE,
                 "loss_history": (hist if CFG.LOG_LOSS_HISTORY else None),
                 "lambda_history": (models[0].get("_lambda_history")
                                    if CFG.LOG_LOSS_HISTORY else None),
                 "scorer_strict": CFG.SCORER_STRICT,
                 "scorer_clause": CFG.SCORER_CLAUSE,
                 "scorer_negation": CFG.SCORER_NEGATION,
                 "degenerate_check": CFG.DEGENERATE_CHECK,
                 "models": [m["id"] for m in models]}
        entry_full = {**entry, "probe_log": ev["rows"]}
        _atomic_write_json(_ex_done_path(key), entry_full)

        manifest.append(entry); results.append(entry_full)
        used += 1
        _write_clean_gate_report(gate_stats)
        _write_rollups(manifest, results); sync_to_drive()

        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    vlog(f"[craft] scanned {scanned} records for {used} usable examples.", 1)
    _write_clean_gate_report(gate_stats)
    _write_rollups(manifest, results); sync_to_drive()
    if state["models"] is not None:
        for m in state["models"]:
            free_model_obj(m["model"])
    return manifest


def main():
    print(f"[v18.4] white-box spatial-lie attack | APGD | models={CFG.MODEL_IDS}")
    print(f"      v18.4 open-gap fixes: CVaR uniform (alpha_final={CFG.CVAR_ALPHA_FINAL}), "
          f"free-run ON gently (sweep={CFG.FREERUN_SWEEP}), "
          f"+{len(OPEN_TEMPLATES_TRAIN_EXTRA)} train-only open templates, "
          f"DRO_SUBSET={CFG.DRO_SUBSET}")
    print(f"      IDEA1 multi_ref={CFG.MULTI_REF} forced_choice={CFG.ADD_FORCED_CHOICE} "
          f"max_refs={CFG.MAX_OPEN_REFS} softmin_tau={CFG.SOFTMIN_TAU}")
    print(f"      OPT1  margin={CFG.MARGIN} ce_w={CFG.CE_WEIGHT} marg_w={CFG.MARGIN_WEIGHT} "
          f"pivot_only={CFG.PIVOT_ONLY} | contrast_w={CFG.CONTRAST_WEIGHT} m={CFG.CONTRAST_MARGIN}")
    print(f"      OPT-A evasion_w={CFG.EVASION_WEIGHT} evasion_margin={CFG.EVASION_MARGIN} "
          f"n_words={len(CFG.EVASION_WORDS)}  (suppress escape routes on open probes)")
    print(f"      OPT-B hardprobe={CFG.HARDPROBE_WEIGHTING} tau={CFG.HARDPROBE_TAU} "
          f"beta={CFG.HARDPROBE_BETA}  (weight the probe the model fails worst)")
    print(f"      OPT-C freerun_w={CFG.FREERUN_WEIGHT} freerun_margin={CFG.FREERUN_MARGIN} "
          f"sweep={CFG.FREERUN_SWEEP or 'off'}")
    print(f"            (the v15 sweep's value is HETEROGENEITY, not its mean: "
          f"fr=0.5 averaged worst yet won 7/20 examples)")
    print(f"      v16   appear_frame={CFG.ADD_APPEAR_FRAME} yesno_contrast={CFG.YESNO_CONTRAST} "
          f"multi_ref_yesno={CFG.MULTI_REF_YESNO} filler_evasion={CFG.ADD_FILLER_EVASION}")
    print(f"            (all default OFF: measured -0.190 SBC together on a "
          f"same-seed controlled A/B)")
    print(f"      v17   sweep_mode={CFG.SWEEP_MODE} restarts={CFG.RESTART_SWEEP} "
          f"early_exit={CFG.SWEEP_EARLY_EXIT} compensate_steps={CFG.COMPENSATE_STEPS} "
          f"variants={n_train_variants()}")
    print(f"      IDEA2 onpolicy={CFG.ONPOLICY} every={CFG.ONPOLICY_EVERY} "
          f"start_frac={CFG.ONPOLICY_START_FRAC} prefix_tok={CFG.ONPOLICY_PREFIX_TOKENS} "
          f"buffer={CFG.ONPOLICY_BUFFER}")
    print(f"            (mint rows from the model's OWN wrong prefix -> optimise "
          f"recovery onto the lie; targets the 'next to' dodges)")
    # ---- v18.7 banner. Printed BEFORE the run so a misconfigured arm is
    # caught at second zero rather than after an hour of GPU time.
    print(f"      IDEA-A probe_lagrangian={CFG.PROBE_LAGRANGIAN}"
          + (f"  C={CFG.PROBE_C} lr={CFG.PROBE_LAMBDA_LR} "
             f"lam0={CFG.PROBE_LAMBDA_INIT} lam_max={CFG.PROBE_LAMBDA_MAX} "
             f"normalize={CFG.PROBE_LAMBDA_NORMALIZE} "
             f"cert_frac={CFG.PROBE_CERT_FRAC}" if CFG.PROBE_LAGRANGIAN else ""))
    if CFG.PROBE_LAGRANGIAN:
        print("            (min||d|| s.t. L_p <= C for EVERY probe; the dual "
              "price of a satisfied probe decays and the budget moves to the "
              "violated ones)")
        if CFG.DRO_CVAR or CFG.HARDPROBE_WEIGHTING or CFG.FAMILY_REWEIGHT:
            print("            [WARN] DRO_CVAR / HARDPROBE_WEIGHTING / "
                  "FAMILY_REWEIGHT are set but are BYPASSED by the Lagrangian. "
                  "Their recorded values in results.json are inert. Use the "
                  "`lagr` preset to turn them off explicitly so the ablation "
                  "table is not misleading.")
        print("            CHECK AFTER THE RUN: probe_certificates.lambda_cv. "
              "Below 0.05 the duals never differentiated between probes and "
              "IDEA-A was effectively uniform -- do NOT attribute SBC to it.")
    print(f"      MBPD  loss={getattr(CFG, 'MBPD_LOSS', False)} alpha_cap={getattr(CFG, 'MBPD_ALPHA_CAP', None)} "
          f"frag_w={getattr(CFG, 'MBPD_FRAG_W', None)} frag_p0={getattr(CFG, 'MBPD_FRAG_P0', None)} "
          f"frag_s={getattr(CFG, 'MBPD_FRAG_S', None)} diag={getattr(CFG, 'MBPD_DIAG', False)}")
    print(f"      CCT   contrast={getattr(CFG, 'CF_CONTRAST', False)} "
          f"alpha={getattr(CFG, 'CF_CONTRAST_ALPHA', None)} beta={getattr(CFG, 'CF_CONTRAST_BETA', None)} "
          f"js={getattr(CFG, 'CF_CONTRAST_JS', None)} js_ref={getattr(CFG, 'CF_CONTRAST_JS_REF', None)} "
          f"replace_gate={getattr(CFG, 'CF_CONTRAST_REPLACE_GATE', None)} "
          f"open_swing={getattr(CFG, 'CF_CONTRAST_OPEN', None)}")
    print(f"      LOSS  mode={getattr(CFG, 'LOSS_MODE', 'classic')} "
          f"cf_families={getattr(CFG, 'CF_FAMILIES', None)} cf_weight={getattr(CFG, 'CF_WEIGHT', None)} "
          f"cf_temp={getattr(CFG, 'CF_TEMP', None)} cf_require_gate={getattr(CFG, 'CF_REQUIRE_GATE', None)} "
          f"cf_kinds={getattr(CFG, 'CF_KINDS', None)} only_families={getattr(CFG, 'ONLY_FAMILIES', None)}")
    print(f"      IDEA-B crn={CFG.CRN_ENABLE} strict_probes={CFG.CRN_STRICT_PROBES} "
          f"stream_seed=0x{int(CFG.CRN_STREAM_SEED):X}")
    if CFG.CRN_ENABLE:
        print("            (random init, EOT dithers and probe subsets are "
              "pinned to (key,salt,phase,step,draw) -- NOT to stream position, "
              "so a refresh call in one arm cannot desynchronise the other)")
        print("            CHECK AFTER THE RUN: two paired arms must print "
              "IDENTICAL [crn] fingerprints and the same crn.draws count. "
              "If they differ, the arms were NOT paired and the delta is noise.")
    else:
        print("            [WARN] CRN OFF. Results 79/81/82 (same config, same "
              "seed) differed by 0.029 SBC and +-0.286 per example under these "
              "conditions. Any ablation delta below ~0.06 is unattributable.")
    print(f"      IDEA3 pool={CFG.PARAPHRASE_POOL} cvar={CFG.DRO_CVAR} "
          f"alpha_final={CFG.CVAR_ALPHA_FINAL} anneal={CFG.CVAR_ANNEAL} "
          f"subset={CFG.DRO_SUBSET}")
    _eff = ("uniform" if (not CFG.DRO_CVAR and not CFG.HARDPROBE_WEIGHTING)
            else "softmax_ema" if not CFG.DRO_CVAR
            else "uniform(alpha=1)" if (float(CFG.CVAR_ALPHA_FINAL) >= 1.0
                                        and not CFG.CVAR_ADAPTIVE)
            else "cvar")
    print(f"      v18.5 CVaR: effective={_eff}  adaptive={CFG.CVAR_ADAPTIVE}"
          f"(delta={CFG.CVAR_ADAPT_DELTA},floor={CFG.CVAR_ALPHA_FLOOR}) "
          f"calibrate={CFG.CVAR_CALIBRATE} typed={CFG.CVAR_TYPED} "
          f"mix={CFG.CVAR_MIX} coldstart_inf={CFG.CVAR_COLDSTART_INF} "
          f"boost_mode={getattr(CFG, 'ONPOLICY_BOOST_MODE', 'ema')}")
    print(f"      v18.6 FMR : family_reweight={CFG.FAMILY_REWEIGHT} "
          f"eps={CFG.FAMILY_RW_EPS} beta={CFG.FAMILY_RW_BETA} "
          f"clamp=[{CFG.FAMILY_RW_MIN}, {CFG.FAMILY_RW_MAX}]")
    if CFG.FAMILY_REWEIGHT and not CFG.ONPOLICY:
        print("      WARNING: FAMILY_REWEIGHT=True but ONPOLICY=False -- no "
              "family failure rates are ever measured, so FMR is inert and "
              "this arm is identical to its base weighting.")
    # The v18.4 postmortem: a run labelled "no CVaR" that had DRO_CVAR=False
    # did NOT get uniform weights, because HARDPROBE_WEIGHTING defaults True
    # and _probe_weights falls through to the softmax branch. That comparison
    # measured CVaR vs soft reweighting, not CVaR vs uniform. Say so loudly.
    if (not CFG.DRO_CVAR) and CFG.HARDPROBE_WEIGHTING:
        print("      WARNING: DRO_CVAR=False with HARDPROBE_WEIGHTING=True is "
              "NOT uniform weighting -- _probe_weights uses the softmax-EMA "
              "branch. For a true uniform control use PRESET='cvar_100' (which "
              "is alpha=1 == uniform) or set HARDPROBE_WEIGHTING=False.")
    if CFG.CVAR_ADAPTIVE and not CFG.ONPOLICY:
        print("      WARNING: CVAR_ADAPTIVE=True but ONPOLICY=False -- the "
              "failure fraction is never measured, so alpha falls back to "
              "CVAR_ALPHA_FINAL and idea (i) is inert.")
    if CFG.CVAR_CALIBRATE and not CFG.ONPOLICY:
        print("      WARNING: CVAR_CALIBRATE=True but ONPOLICY=False -- no "
              "(loss, outcome) pairs are ever observed, so the calibrator "
              "never reaches CVAR_CAL_MIN_OBS and ranking stays on raw EMA.")
    if CFG.PARAPHRASE_POOL:
        describe_pool_split()
    print(f"      GATE  mint_gate={CFG.ONPOLICY_MINT_GATE} "
          f"require_subject={CFG.MINT_REQUIRE_SUBJECT} "
          f"reject_degen={CFG.MINT_REJECT_DEGEN}")
    print(f"            (only mint on-policy rows from REAL dodges, not from "
          f"hallucinated scenes)")
    print(f"      MASK  enable={CFG.MASK_ENABLE} mode={CFG.MASK_MODE} "
          f"keep_frac={CFG.MASK_KEEP_FRAC} blur={CFG.MASK_BLUR_SIGMA} "
          f"dilate={CFG.MASK_DILATE}")
    if CFG.MASK_ENABLE:
        print(f"            (freeze {100*(1-CFG.MASK_KEEP_FRAC):.0f}% of pixels -> "
              f"the scene cannot be replaced; expect SBC DOWN, preservation UP)")
    print(f"      CURVE mode={CFG.CURVE_MODE}"
          + (f" values={[round(v*255,1) for v in CFG.EPS_CURVE_VALUES]}/255 "
             f"ref={CFG.CURVE_REFERENCE*255:.0f}/255" if CFG.CURVE_MODE == "eps"
             else (f" values={CFG.MASK_CURVE_VALUES}" if CFG.CURVE_MODE == "mask" else "")))
    print(f"      SCORE strict={CFG.SCORER_STRICT} clause={CFG.SCORER_CLAUSE} "
          f"negation={CFG.SCORER_NEGATION} degen_check={CFG.DEGENERATE_CHECK} "
          f"audit={CFG.SCORER_AUDIT}")
    selftest_scorer()
    print(f"      OPT3  quant_ste={CFG.QUANT_STE} dither={CFG.EOT_DITHER*255:.2f}LSB")
    print(f"      OPT4  steps_by_family={CFG.STEPS_BY_FAMILY} eot={CFG.EOT_SAMPLES} "
          f"antithetic={CFG.EOT_ANTITHETIC}  (dialed back for n>=300)")
    print(f"      MEM   max_rows_per_fwd={CFG.MAX_ROWS_PER_FWD} grad_ckpt={CFG.GRAD_CHECKPOINT} "
          f"alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF','')}")
    print(f"      eps={CFG.EPS:.4f} prompt_eot={CFG.PROMPT_EOT} out={CFG.OUTPUT_DIR}")
    t_all = time.time()
    craft_phase()
    print(f"\n[done] complete in {time.time()-t_all:.1f}s. Outputs in {CFG.OUTPUT_DIR}.")


def run_step3_suite(arms=None, seeds=None, base_out_dir="/content/drive/MyDrive/v1812/step3_baselines"):
    """Step 3 runner: executes any list of (arm, seed) pairs sequentially, saving each
    into its own dedicated subdirectory `<base_out_dir>/<arm>__seed<seed>` and
    printing the consolidated Step 3 comparison table across all completed arms."""
    if arms is None:
        arms = list(getattr(CFG, "RUN_SUITE", None) or [getattr(CFG, "BASELINE_ARM", "full")])
    if seeds is None:
        seeds = [int(CFG.SEED)]
    summary_rows = []
    for s in seeds:
        for arm in arms:
            CFG.SEED = int(s)
            CFG.BASELINE_ARM = str(arm)
            CFG.PRESET = "reclaim" if str(arm) == "full" else str(arm)
            _apply_preset()
            CFG.BASELINE_ARM = str(arm)
            if len(arms) > 1 or len(seeds) > 1 or str(arm) != "full":
                CFG.OUTPUT_DIR = os.path.join(base_out_dir, f"{arm}__seed{s}")
            print(f"\n======================================================================")
            print(f"  STEP 3 RUN: ARM={arm!r} | SEED={s} | OUT={CFG.OUTPUT_DIR}")
            print(f"======================================================================")
            main()
            mpath = os.path.join(CFG.OUTPUT_DIR, "manifest.json")
            if os.path.exists(mpath):
                mdf = pd.DataFrame(_read_json(mpath))
                if len(mdf) > 0:
                    summary_rows.append({
                        "arm": arm,
                        "seed": s,
                        "n": len(mdf),
                        "coherence_15": round(float(mdf["coherence"].mean()), 3),
                        "SBC_15": round(float(mdf["SBC"].mean()), 3),
                        "coherence_7": round(float(mdf["coherence_7"].mean()), 3) if "coherence_7" in mdf else None,
                        "SBC_7": round(float(mdf["SBC_7"].mean()), 3) if "SBC_7" in mdf else None,
                        "sbc_open": round(float(mdf["sbc_open"].mean()), 3),
                        "sbc_yes": round(float(mdf["sbc_yes"].mean()), 3),
                        "sbc_no": round(float(mdf["sbc_no"].mean()), 3),
                        "preserve_f1": round(float(mdf["preserve_f1"].mean()), 3),
                        "lpips": round(float(mdf["lpips"].dropna().mean()), 4) if "lpips" in mdf and mdf["lpips"].notna().any() else None,
                        "time_s": round(float(mdf["wall_time_s"].dropna().mean()), 1) if "wall_time_s" in mdf and mdf["wall_time_s"].notna().any() else None,
                    })
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        os.makedirs(base_out_dir, exist_ok=True)
        df_sum.to_csv(os.path.join(base_out_dir, "step3_main_table.csv"), index=False)
        print("\n================ STEP 3 MAIN COMPARISON TABLE ================")
        print(df_sum.to_string(index=False))
        print(f"  wrote {os.path.join(base_out_dir, 'step3_main_table.csv')}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Coherent Spatial-Lie Attack on VLMs (Step 2 + Step 3)")
    parser.add_argument("--arm", type=str, default=None,
                        help="Single arm to run: full | B0 | B1 | B2 | B3 | B4 | B5")
    parser.add_argument("--suite", type=str, nargs="*", default=None,
                        help="Multiple arms to run in sequence, e.g. --suite B0 B1 B2 B3 B4 B5 full")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds to run, e.g. --seeds 0 1 2")
    parser.add_argument("--num_examples", type=int, default=None,
                        help="Override CFG.NUM_EXAMPLES (default 200)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Override CFG.OUTPUT_DIR")
    args, _unknown = parser.parse_known_args()
    if args.num_examples is not None:
        CFG.NUM_EXAMPLES = int(args.num_examples)
    if args.out_dir is not None:
        CFG.OUTPUT_DIR = str(args.out_dir)
    if args.suite is not None or args.seeds is not None:
        run_step3_suite(arms=args.suite, seeds=args.seeds)
    elif args.arm is not None:
        CFG.BASELINE_ARM = str(args.arm)
        if str(args.arm) != "full":
            CFG.PRESET = str(args.arm)
            _apply_preset()
            CFG.BASELINE_ARM = str(args.arm)
            if args.out_dir is None:
                CFG.OUTPUT_DIR = f"/content/drive/MyDrive/v1812/step3_baselines/{args.arm}__seed{CFG.SEED}"
        main()
    elif getattr(CFG, "RUN_SUITE", None):
        run_step3_suite(arms=CFG.RUN_SUITE)
    else:
        main()