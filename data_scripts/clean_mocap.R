# ---- Setup ----
library(dplyr)
library(readr)
library(stringr)

# Adjust these:
INPUT_DIR  <- "MOCAP_SENSOR_1003/teresa/mocap"   # folder to scan for .csv
OUTPUT_DIR <- "MOCAP_SENSOR_1003/teresa/mocap/cleaned"    # where cleaned files go
RECURSIVE  <- TRUE                              # set FALSE if not needed

# Make sure output dir exists
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ---- Constants from your script ----
drop_cols <- unique(c(
  "dancepad:Marker 001", "dancepad:Marker 0010", "dancepad:Marker 002",
  "dancepad:Marker 004", "dancepad:Marker 007", "dancepad:Marker 008",
  "dancepad:Marker 009", "dancepad:Marker 011",
  "dancepad", paste0("dancepad.", 1:6),
  "dancepad.Marker.001", "dancepad.Marker.001.1", "dancepad.Marker.001.2",
  "dancepad.Marker.002", "dancepad.Marker.002.1", "dancepad.Marker.002.2",
  "dancepad.Marker.003", "dancepad.Marker.003.1", "dancepad.Marker.003.2",
  "dancepad.Marker.004", "dancepad.Marker.004.1", "dancepad.Marker.004.2",
  "dancepad.Marker.005", "dancepad.Marker.005.1", "dancepad.Marker.005.2",
  "dancepad.Marker.006", "dancepad.Marker.006.1", "dancepad.Marker.006.2",
  "dancepad.Marker.007", "dancepad.Marker.007.1", "dancepad.Marker.007.2",
  "dancepad.Marker.008", "dancepad.Marker.008.1", "dancepad.Marker.008.2",
  "dancepad.Marker.009", "dancepad.Marker.009.1", "dancepad.Marker.009.2",
  "dancepad.Marker.0010", "dancepad.Marker.0010.1", "dancepad.Marker.0010.2",
  "dancepad.Marker.011", "dancepad.Marker.011.1", "dancepad.Marker.011.2",
  "dancepad.Marker.001.3", "dancepad.Marker.001.4", "dancepad.Marker.001.5",
  "dancepad.Marker.0010.3", "dancepad.Marker.0010.4", "dancepad.Marker.0010.5",
  "dancepad.Marker.002.3", "dancepad.Marker.002.4", "dancepad.Marker.002.5",
  "dancepad.Marker.004.3", "dancepad.Marker.004.4", "dancepad.Marker.004.5",
  "dancepad.Marker.007.3", "dancepad.Marker.007.4", "dancepad.Marker.007.5",
  "dancepad.Marker.008.3", "dancepad.Marker.008.4", "dancepad.Marker.008.5",
  "dancepad.Marker.009.3", "dancepad.Marker.009.4", "dancepad.Marker.009.5",
  "dancepad.Marker.011.3", "dancepad.Marker.011.4", "dancepad.Marker.011.5"
))

joint_map <- c(
  "Skeleton.004.Skeleton.004" = "base",
  "Skeleton.001.Skeleton.001" = "base",
  "Skeleton.Skeleton" = "base",
  "skeleton.skeleton" = "base",
  "Skeleton.Ab"       = "abdomen",
  "Skeleton.Chest"    = "chest",
  "Skeleton.Neck"     = "neck",
  "Skeleton.Head"     = "head",
  "Skeleton.LShoulder" = "left_shoulder",
  "Skeleton.RShoulder" = "right_shoulder",
  "Skeleton.LUArm"     = "left_upper_arm",
  "Skeleton.LUPA"      = "left_upper_arm",
  "Skeleton.LFArm"     = "left_forearm",
  "Skeleton.LFRM"      = "left_forearm",
  "Skeleton.RUArm"     = "right_upper_arm",
  "Skeleton.RUPA"      = "right_upper_arm",
  "Skeleton.RFArm"     = "right_forearm",
  "Skeleton.RFRM"      = "right_forearm",
  "Skeleton.LELB"      = "left_elbow",
  "Skeleton.RELB"      = "right_elbow",
  "Skeleton.LSHO"      = "left_shoulder",
  "Skeleton.RSHO"      = "right_shoulder",
  "Skeleton.LWRB"      = "left_wrist_b",
  "Skeleton.LWRA"      = "left_wrist_a",
  "Skeleton.RWRB"      = "right_wrist_b",
  "Skeleton.RWRA"      = "right_wrist_a",
  "Skeleton.LHand"     = "left_hand",
  "Skeleton.RHand"     = "right_hand",
  "Skeleton.LThumb1"   = "left_thumb1",
  "Skeleton.LThumb2"   = "left_thumb2",
  "Skeleton.LThumb3"   = "left_thumb3",
  "Skeleton.LIndex1"   = "left_index1",
  "Skeleton.LIndex2"   = "left_index2",
  "Skeleton.LIndex3"   = "left_index3",
  "Skeleton.LMiddle1"  = "left_middle1",
  "Skeleton.LMiddle2"  = "left_middle2",
  "Skeleton.LMiddle3"  = "left_middle3",
  "Skeleton.LRing1"    = "left_ring1",
  "Skeleton.LRing2"    = "left_ring2",
  "Skeleton.LRing3"    = "left_ring3",
  "Skeleton.LPinky1"   = "left_pinky1",
  "Skeleton.LPinky2"   = "left_pinky2",
  "Skeleton.LPinky3"   = "left_pinky3",
  "Skeleton.RThumb1"   = "right_thumb1",
  "Skeleton.RThumb2"   = "right_thumb2",
  "Skeleton.RThumb3"   = "right_thumb3",
  "Skeleton.RIndex1"   = "right_index1",
  "Skeleton.RIndex2"   = "right_index2",
  "Skeleton.RIndex3"   = "right_index3",
  "Skeleton.RMiddle1"  = "right_middle1",
  "Skeleton.RMiddle2"  = "right_middle2",
  "Skeleton.RMiddle3"  = "right_middle3",
  "Skeleton.RRing1"    = "right_ring1",
  "Skeleton.RRing2"    = "right_ring2",
  "Skeleton.RRing3"    = "right_ring3",
  "Skeleton.RPinky1"   = "right_pinky1",
  "Skeleton.RPinky2"   = "right_pinky2",
  "Skeleton.RPinky3"   = "right_pinky3",
  "Skeleton.LThigh"    = "left_thigh",
  "Skeleton.RThigh"    = "right_thigh",
  "Skeleton.LShin"     = "left_shin",
  "Skeleton.RShin"     = "right_shin",
  "Skeleton.LFoot"     = "left_foot",
  "Skeleton.RFoot"     = "right_foot",
  "Skeleton.LToe"      = "left_toe_marker",
  "Skeleton.RToe"      = "right_toe_marker",
  "Skeleton.LANK"      = "left_ankle",
  "Skeleton.RANK"      = "right_ankle",
  "Skeleton.LHEE"      = "left_heel",
  "Skeleton.RHEE"      = "right_heel",
  "Skeleton.LFHD"      = "left_forehead",
  "Skeleton.LBHD"      = "left_back_head",
  "Skeleton.RFHD"      = "right_forehead",
  "Skeleton.RBHD"      = "right_back_head",
  "Skeleton.C7"        = "7_cervical_vertebra",
  "Skeleton.T10"       = "t10",
  "Skeleton.RBAK"      = "right_back",
  "Skeleton.CLAV"      = "clavicle",
  "Skeleton.STRN"      = "sternum",
  "Skeleton.LASI"      = "left_asi",
  "Skeleton.LPSI"      = "left_psi",
  "Skeleton.RASI"      = "right_asi",
  "Skeleton.RPSI"      = "right_psi",
  "Skeleton.RFIN" = "right_finger",
  "Skeleton.LFIN" = "left_finger",
  "Skeleton.LTHI" = "left_thigh",
  "Skeleton.LKNE" = "left_knee",
  "Skeleton.LTIB" = "left_tibia",
  "Skeleton.RTIB" = "right_tibia",
  "Skeleton.RToe" = "right_toe",
  "Left.Foot" = "left_foot",
  "Right.Foot" = "right_foot",
  "Skeleton.LTOE" = "left_toe",
  "Skeleton.RTHI" = "right_thigh",
  "Skeleton.RKNE" = "right_knee",
  "Skeleton.RTOE" = "right_toe"
)

gsub("Skeleton\\.", "Skeleton.001.", joint_map)


# ---- Core processor for ONE file ----
process_mocap_file <- function(in_csv, out_dir) {
  message("Processing: ", in_csv)
  # read: original files have 3 metadata rows to skip
  mocap_data <- suppressMessages(readr::read_csv(in_csv, skip = 3, show_col_types = FALSE)) %>% as.data.frame()

  # drop first row (your original code removed it)
  if (nrow(mocap_data) >= 1) mocap_data <- mocap_data[-1, , drop = FALSE]

  # remove unwanted columns if present
  keep <- setdiff(names(mocap_data), drop_cols)
  mocap_data <- mocap_data[, keep, drop = FALSE]

  # original colnames
  original_joint_names <- names(mocap_data)

  # clean suffixes like ".1", ".6"
  clean_joint_names <- sub("\\.[0-9]+$", "", original_joint_names)

  # replacement using dictionary (fall back to original)
  renamed_joint_names <- ifelse(
    clean_joint_names %in% names(joint_map),
    unname(joint_map[clean_joint_names]),
    original_joint_names
  )

  # axis row is row 2 in your original (after the top skip)
  # but we dropped one row above; the "axis" metadata is now in row 2 of current df?
  # Your code: axis <- unlist(mocap_data[2, ])
  # Keep consistent with that:
  if (nrow(mocap_data) < 2) stop("Not enough rows to read axis row in: ", in_csv)
  axis <- unlist(mocap_data[2, ], use.names = FALSE)

  new_colnames <- paste(renamed_joint_names, axis, sep = "_")
  new_colnames <- gsub("NA", "", new_colnames)
  new_colnames <- gsub("__", "_", new_colnames)

  # remove the two metadata rows
  if (nrow(mocap_data) < 3) stop("Not enough rows after metadata in: ", in_csv)
  mocap_data <- mocap_data[-c(1:2), , drop = FALSE]
  names(mocap_data) <- new_colnames

  # standardize required columns
  if (ncol(mocap_data) >= 2) {
    names(mocap_data)[1] <- "Frame"
    names(mocap_data)[2] <- "time_ms"
  }
  # force ms numeric (original *1000)
  if ("time_ms" %in% names(mocap_data)) {
    mocap_data$time_ms <- suppressWarnings(as.numeric(mocap_data$time_ms) * 1000)
  }

  # ensure base_X as in your code (guard if missing)
  if (ncol(mocap_data) >= 3) names(mocap_data)[3] <- "base_X"

  # write out with mirrored relative path
  rel <- fs::path_rel(in_csv, start = INPUT_DIR)
  out_file <- fs::path(OUT_DIR <- out_dir,
                       fs::path_ext_set(rel, "csv")) # ensure .csv
  fs::dir_create(fs::path_dir(out_file))
  readr::write_csv(mocap_data, out_file)

  message("Saved -> ", out_file)
  invisible(out_file)
}

# ---- Walk the repository and process everything ----
csvs <- list.files(INPUT_DIR, pattern = "\\.csv$", full.names = TRUE, recursive = RECURSIVE)

if (length(csvs) == 0) {
  message("No CSV files found in: ", INPUT_DIR)
} else {
  # Helpful dependency used above
  if (!requireNamespace("fs", quietly = TRUE)) stop("Please install.packages('fs')")
  out_paths <- lapply(csvs, function(p) {
    tryCatch(process_mocap_file(p, OUTPUT_DIR),
             error = function(e) { message("❌ ", conditionMessage(e)); NULL })
  })
  message("Done. Wrote ", sum(!sapply(out_paths, is.null)), " files to: ", OUTPUT_DIR)
}