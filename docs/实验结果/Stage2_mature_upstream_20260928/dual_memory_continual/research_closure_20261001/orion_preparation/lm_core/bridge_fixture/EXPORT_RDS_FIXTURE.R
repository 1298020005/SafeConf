args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
source_root <- args[[1L]]
output <- args[[2L]]
dir.create(output, recursive = TRUE)
for (context in c("HCT116", "HEK293T")) {
  X <- readRDS(file.path(source_root, paste0("SYNTHETIC_", context), "TRAIN_X.rds"))
  stopifnot(nrow(X) == 38606L, ncol(X) == 24L)
  writeBin(as.vector(X), file.path(output, paste0(context, ".f64le")), size = 8L, endian = "little")
  saveRDS(X, file.path(output, paste0(context, "_REFERENCE.rds")), compress = FALSE)
}
cat("SYNTHETIC_RDS_EXPORT_PASS\n")
