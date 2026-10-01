args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
repo <- normalizePath(args[[1L]])
root <- normalizePath(args[[2L]])
cli <- file.path(repo, "tools/scripts/run_safeconf_orion_lm_blind.R")
source(cli)
original <- read_tsv(file.path(root, "inputs/FIT_MANIFEST.tsv"), "context_id")
row <- as.list(original[1L, , drop = FALSE])
checks <- list()
for (case in c("query_truth_column", "validation_in_TRAIN", "missing_actual_permit")) {
  folder <- file.path(root, paste0("guard_", case))
  dir.create(folder)
  bad <- row
  bad$train_x <- file.path(folder, "POISON_TRAIN_X.rds")
  writeLines("Invalid RDS: numeric training values must not be loaded", bad$train_x)
  r <- read_tsv(row$receipt, c("key", "value"))
  r$value[r$key == "train_x_sha256"] <- sha256(bad$train_x)
  if (case == "query_truth_column") {
    bad$queries <- file.path(folder, "QUERY_WITH_FORBIDDEN_COLUMN.tsv")
    q <- read_tsv(row$queries, "query_id")
    q$forbidden_query_expression <- "synthetic forbidden field"
    write_tsv(q, bad$queries)
    r$value[r$key == "queries_sha256"] <- sha256(bad$queries)
    expected <- "Query file must contain only identity metadata"
  } else if (case == "validation_in_TRAIN") {
    bad$conditions <- file.path(folder, "ILLEGAL_CONDITIONS.tsv")
    c <- read_tsv(row$conditions, "role")
    c$role[[1L]] <- "VALIDATION"
    write_tsv(c, bad$conditions)
    r$value[r$key == "conditions_sha256"] <- sha256(bad$conditions)
    expected <- "Illegal expression role or missing own-context NTC"
  } else {
    r$value[r$key == "dataset_kind"] <- "AUTHORIZED_TRAIN"
    expected <- "Required bound authorization artifact: expression_permit"
  }
  bad$receipt <- file.path(folder, "INPUT_RECEIPT.tsv")
  write_tsv(r, bad$receipt)
  manifest <- file.path(folder, "MANIFEST.tsv")
  write_tsv(as.data.frame(bad), manifest)
  log <- file.path(folder, "REJECTION.log")
  exit <- system2(file.path(R.home("bin"), "Rscript"), shQuote(c("--vanilla", cli, "--mode", "fit",
       "--manifest", manifest, "--output-dir", file.path(folder, "outputs"))), stdout = log, stderr = log)
  stopifnot(exit != 0L, any(grepl(expected, readLines(log), fixed = TRUE)))
  checks[[case]] <- data.frame(check = paste0(case, "_rejected_before_numeric_TRAIN_load"), pass = TRUE)
}
write_tsv(do.call(rbind, checks), file.path(root, "INPUT_GUARD_COMPARISON.tsv"))
cat("BLIND_INPUT_GUARDS_PASS", length(checks), "checks\n")
