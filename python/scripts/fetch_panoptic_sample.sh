#!/usr/bin/env bash
# fetch_panoptic_sample.sh -- minimal CMU Panoptic Studio fetch for MonoArm validation
#
# Downloads ONLY what evaluate_panoptic.py needs for one sequence and one HD
# camera: the coco19 3D body keypoints plus a single HD video, then extracts
# frames in the naming convention the evaluator expects
# (hdImgs/<cam>/<cam>_%08d.jpg).
#
# The upstream getData.sh defaults to all 31 HD views; this fetches one, which
# is roughly a thirtieth of the bandwidth.
#
# CMU Panoptic Studio is NOT registration-gated -- it is "freely available for
# non-commercial and research purpose only", conditional on citing the dataset
# paper (Joo et al., ICCV 2015). Run this on a machine with unrestricted
# outbound network access; it will NOT work from a sandbox whose egress policy
# blocks the data host.
#
# Usage:
#   ./fetch_panoptic_sample.sh [sequence] [camera] [outdir]
#   ./fetch_panoptic_sample.sh 171204_pose1_sample 00_00 ./data/panoptic
#   PANOPTIC_ENDPOINT=http://vcl.snu.ac.kr/panoptic ./fetch_panoptic_sample.sh   # SNU mirror
set -euo pipefail

SEQ="${1:-171204_pose1_sample}"
CAM="${2:-00_00}"
OUTDIR="${3:-./data/panoptic}"
ENDPOINT="${PANOPTIC_ENDPOINT:-http://domedb.perception.cs.cmu.edu}"

command -v ffmpeg >/dev/null 2>&1 || { echo "ERROR: ffmpeg is required (apt install ffmpeg / brew install ffmpeg)"; exit 1; }
command -v curl   >/dev/null 2>&1 || { echo "ERROR: curl is required"; exit 1; }

DEST="${OUTDIR}/${SEQ}"
mkdir -p "${DEST}/hdVideos"
cd "${DEST}"

echo "[1/4] 3D body keypoints (coco19)"
if [ ! -f hdPose3d_stage1_coco19.tar ]; then
  curl -C - -f -o hdPose3d_stage1_coco19.tar \
    "${ENDPOINT}/webdata/dataset/${SEQ}/hdPose3d_stage1_coco19.tar"
fi
[ -d hdPose3d_stage1_coco19 ] || tar -xf hdPose3d_stage1_coco19.tar

echo "[2/4] HD video for camera ${CAM}"
if [ ! -f "hdVideos/hd_${CAM}.mp4" ]; then
  curl -C - -f -o "hdVideos/hd_${CAM}.mp4" \
    "${ENDPOINT}/webdata/dataset/${SEQ}/videos/hd_shared_crf20/hd_${CAM}.mp4"
fi

echo "[3/4] Extracting frames (this is the slow step)"
mkdir -p "hdImgs/${CAM}"
if [ -z "$(ls -A "hdImgs/${CAM}" 2>/dev/null)" ]; then
  # -start_number 0 and the %08d pattern match evaluate_panoptic.py's
  # expected filenames exactly.
  ffmpeg -loglevel warning -i "hdVideos/hd_${CAM}.mp4" -q:v 1 -f image2 \
    -start_number 0 "hdImgs/${CAM}/${CAM}_%08d.jpg"
fi

N_GT=$(ls hdPose3d_stage1_coco19/ | wc -l)
N_IMG=$(ls "hdImgs/${CAM}/" | wc -l)
echo "[4/4] Done: ${N_GT} ground-truth frames, ${N_IMG} extracted images"
echo
echo "Now run the evaluation from the python/ directory:"
echo
echo "    python -m scripts.evaluate_panoptic \\"
echo "        --sequence_dir $(pwd) \\"
echo "        --camera ${CAM} \\"
echo "        --frameworks mediapipe movenet_lightning"
echo
echo "Note: the evaluator runs a correlation-based temporal-alignment check"
echo "first -- Panoptic's video and 3D reconstruction are separate pipelines"
echo "with no guaranteed common start frame, so pairing them by index alone"
echo "can be silently wrong. Check the reported shift is small and its"
echo "correlation is high before trusting any accuracy number."
