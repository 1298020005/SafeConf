args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
repo <- normalizePath(args[[1L]]); root <- normalizePath(args[[2L]])
source(file.path(repo, "tools/scripts/run_safeconf_orion_lm_blind.R"))
original <- file.path(root, "bridge/HCT116/TRAIN_X.f64le")
checks <- list()
for (case in c("shape_byte_count", "corrupt_payload_hash")) {
  folder <- file.path(root, paste0("guard_", case)); dir.create(folder)
  binary <- file.path(folder, "TRAIN_X.f64le")
  stopifnot(file.copy(original, binary))
  sidecar <- read_tsv(paste0(original, ".meta.tsv"), c("key", "value"))
  if (case == "shape_byte_count") {
    sidecar$value[sidecar$key == "n_conditions"] <- "25"
    expected <- "Binary byte count does not match shape"
  } else {
    con <- file(binary, "r+b")
    first <- readBin(con, "raw", n = 1L); seek(con, 0L, origin = "start")
    writeBin(as.raw(bitwXor(as.integer(first), 1L)), con); close(con)
    expected <- "Binary payload hash differs"
  }
  write_tsv(sidecar, paste0(binary, ".meta.tsv"))
  result <- tryCatch({read_matrix(binary); "unexpected success"}, error = function(e) conditionMessage(e))
  stopifnot(identical(result, expected))
  checks[[case]] <- data.frame(check = paste0(case, "_rejected_before_readBin_numeric_loading"), pass = TRUE)
}
write_tsv(do.call(rbind, checks), file.path(root, "BINARY_GUARD_COMPARISON.tsv"))
cat("BINARY_FORMAT_GUARDS_PASS", length(checks), "checks\n")
