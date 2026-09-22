#!/usr/bin/env bash
# SpaceNet 7 train download + extract (user approved 2026-09-22). Anonymous S3 HTTPS; sizes: csvs 496MB, train 9.16GB.
set -u
ROOT=/home/work/data/olmoearth/spacenet7; mkdir -p "$ROOT/tarballs"; cd "$ROOT" || exit 1
for f in SN7_buildings_train_csvs.tar.gz SN7_buildings_train.tar.gz; do
  curl -s -o "tarballs/$f" "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN7_buildings/tarballs/$f" && echo "DL_OK $f $(stat -c %s tarballs/$f)"
done
(cd tarballs && sha256sum *.tar.gz > SHA256SUMS)
tar xzf tarballs/SN7_buildings_train_csvs.tar.gz && tar xzf tarballs/SN7_buildings_train.tar.gz && echo EXTRACT_OK
ls | head; echo SN7_DONE
