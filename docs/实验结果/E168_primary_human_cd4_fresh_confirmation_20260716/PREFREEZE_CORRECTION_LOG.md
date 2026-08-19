# E168 pre-freeze correction log

独立审计在任何 freeze commit 和任何 X 访问之前，否决了一版未提交的候选清单：它曾用四位 donor 的 targeting `n_cells` 和 12-combination coverage 作 eligibility。`n_cells` 可能携带扰动后存活信息，故该候选输出已删除，未用于本合同。最终版本用官方 expression-independent guide design/annotation、基因身份轴、scGPT 词表和 label-free guide identity presence 定义 assay-available universe，再做 hash 选择。该 availability eligibility 会如实披露；它不使用 abundance、表达或效应。整个修正过程中 test targeting X、DE、guide efficacy 和 keep_* 值均未读取。
