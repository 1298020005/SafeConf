# References to insert before journal formatting

1. **PertEMA.** OfficialBishal/PertEMA, *Perturbation Error Meta-Assessment*. Official repository: https://github.com/OfficialBishal/PertEMA. The project provides the closest post-hoc error-learning comparison and should be cited for the target-specific error estimator.
2. **GEARS.** Roohani Y, Huang K, Leskovec J. Predicting transcriptional outcomes of novel multigene perturbations with GEARS. *Nature Biotechnology* (2023). Official code: https://github.com/snap-stanford/GEARS.
3. **scGPT.** Cui H et al. scGPT: toward building a foundation model for single-cell multi-omics using generative AI. *Nature Methods* 21, 1470–1480 (2024). https://www.nature.com/articles/s41592-024-02201-0
4. **Replogle GWPS.** Replogle JM et al. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* 185, 2559–2575.e28 (2022), doi:10.1016/j.cell.2022.05.013. The local K562 GWPS resource and its processing contract are documented in the evidence package.
5. **Frangieh perturbation study.** Frangieh M et al. Multi-modal pooled Perturb-CITE-seq screens in patient models define novel mechanisms of cancer immune evasion. The frozen native512 GEARS/scGPT stress analysis is documented in the repository's Frangieh contract.
6. **Evaluation and uncertainty.** PRESCRIBE (Cheng et al., NeurIPS 2025) and the control-reuse evaluation analysis by Nicol et al. (bioRxiv 2026) are included in `REFERENCES.bib`; they motivate the distinction between internal uncertainty, measurement bias, and the external public-evidence layer.

The manuscript must cite the original methods and official implementations. SafeConf should be described as adding a public-evidence cold-start layer and an information-budget evaluation contract, not as replacing GEARS, scGPT, or PertEMA.
