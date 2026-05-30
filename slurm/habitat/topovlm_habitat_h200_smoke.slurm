#!/bin/bash

#SBATCH -p h200
#SBATCH --gres=gpu:h200:1
#SBATCH -J topovlm_habitat_h200_smoke
#SBATCH --time=7-00:00:00
#SBATCH -o /home/heejun/Projects/TopoVLM/artifacts/logs/slurm-%x-%j.out
#SBATCH -e /home/heejun/Projects/TopoVLM/artifacts/logs/slurm-%x-%j.err

set -euo pipefail

if ! mkdir -p "/scratch"/"$USER" 2>/dev/null; then
  echo "[blocker] scratch_root_not_writable path=/scratch/$USER host=$(hostname)" >&2
  exit 34
fi
if [ ! -w "/scratch"/"$USER" ]; then
  echo "[blocker] scratch_root_not_writable path=/scratch/$USER host=$(hostname)" >&2
  exit 34
fi

SOURCE_COMMIT="589e2f4a45c4edfcb36a9c24acf571b0e835427d"
if [[ ! "$SOURCE_COMMIT" =~ ^[0-9a-fA-F]{7,40}$ ]] || ! git -C "/home/heejun/Projects/TopoVLM" cat-file -e "$SOURCE_COMMIT^{commit}"; then
  echo "[blocker] SOURCE_COMMIT is not a valid commit: $SOURCE_COMMIT" >&2
  exit 32
fi

SUBMIT_DIR="/home/heejun/Projects/TopoVLM"
CODE_ROOT="/scratch/$USER/workcache/TopoVLM/code"
CODE_SOURCE="$CODE_ROOT/$SOURCE_COMMIT"
DATA_CACHE="/scratch/$USER/workcache/TopoVLM/data"
RUN_ROOT="/scratch/$USER/workcache/TopoVLM/runs/${SLURM_JOB_ID}"
WORKDIR="$RUN_ROOT/work"
RUN_OUTPUT_DIR="$RUN_ROOT/outputs"
RUN_LOCAL_DIR="$RUN_ROOT/local"
RUN_TMP_DIR="/scratch/$USER/tmp/$SLURM_JOB_ID"
STAGE_PROGRESS_FILE="$RUN_OUTPUT_DIR/stage_progress.json"
STAGE_MANIFEST="$RUN_ROOT/stage_manifest.json"
if [ "${#RUN_TMP_DIR}" -gt 60 ]; then
  echo "[blocker] slurm_tmpdir_too_long_for_unix_sockets path=$RUN_TMP_DIR length=${#RUN_TMP_DIR}" >&2
  exit 39
fi
mkdir -p "$CODE_ROOT" "$DATA_CACHE" "$RUN_ROOT" "$WORKDIR" "$RUN_OUTPUT_DIR" "$RUN_LOCAL_DIR" "$RUN_TMP_DIR"
cat > "$STAGE_MANIFEST" <<'JSON'
{
  "artifact_type": "slurm_stage_manifest",
  "artifacts": [],
  "cache": {
    "code": {
      "commit_path": "/scratch/$USER/workcache/TopoVLM/code/589e2f4a45c4edfcb36a9c24acf571b0e835427d",
      "path": "/scratch/$USER/workcache/TopoVLM/code",
      "sync_delete": false
    },
    "data": {
      "path": "/scratch/$USER/workcache/TopoVLM/data",
      "sync_delete": false
    },
    "output": {
      "path": "$RUN_OUTPUT_DIR",
      "sync_delete": false
    },
    "run": {
      "path": "/scratch/$USER/workcache/TopoVLM/runs/${SLURM_JOB_ID}",
      "sync_delete": false
    }
  },
  "code": {
    "source": "/home/heejun/Projects/TopoVLM",
    "source_commit": "589e2f4a45c4edfcb36a9c24acf571b0e835427d",
    "sync": "git_archive_commit_immutable_code",
    "target": "/scratch/$USER/workcache/TopoVLM/code/589e2f4a45c4edfcb36a9c24acf571b0e835427d",
    "workdir": "$WORKDIR"
  },
  "data": [
    {
      "reason": "HM3D ObjectNav v2 episode shards used by habitat/prismatic_bc_smoke_staged.",
      "source": "/data/topovlm/habitat/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2",
      "target": "topovlm/habitat/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2"
    },
    {
      "reason": "HM3D scene meshes and semantic annotations resolved by Habitat ObjectNav scene ids.",
      "source": "/data/topovlm/habitat/scene_datasets/hm3d",
      "target": "topovlm/habitat/scene_datasets/hm3d"
    },
    {
      "reason": "Frozen Prismatic VLM weights used by the staged TopoVLM config.",
      "source": "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b",
      "target": "topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
    }
  ],
  "flow_key": "topovlm_habitat_h200_smoke_${SLURM_JOB_ID}",
  "generated_at_utc": "2026-05-30T04:46:58+00:00",
  "job_name": "topovlm_habitat_h200_smoke",
  "output": {
    "default_stageout": "/home/heejun/Projects/TopoVLM/artifacts/logs/slurm_outputs/topovlm_habitat_h200_smoke/${SLURM_JOB_ID}",
    "run_output_dir": "$RUN_OUTPUT_DIR",
    "run_root": "/scratch/$USER/workcache/TopoVLM/runs/${SLURM_JOB_ID}",
    "run_tmp_dir": "$RUN_TMP_DIR"
  },
  "schema_version": 1
}
JSON
export WANDB_MODE="disabled"
export WANDB_SILENT=true

write_stage_progress() {
  local phase="$1"
  local stage_status="$2"
  local note="${3:-}"
  local updated_at
  updated_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  mkdir -p "$(dirname "$STAGE_PROGRESS_FILE")"
  printf '{"artifact_type":"slurm_stage_progress","job_name":"%s","job_id":"%s","phase":"%s","status":"%s","updated_at_utc":"%s","note":"%s"}\n' \
    "topovlm_habitat_h200_smoke" "$SLURM_JOB_ID" "$phase" "$stage_status" "$updated_at" "$note" > "$STAGE_PROGRESS_FILE"
  echo "[stage-progress] phase=$phase status=$stage_status note=$note file=$STAGE_PROGRESS_FILE"
}

write_stage_progress "env" "running" "tarball_env"
ENVCACHE_ROOT="/scratch/$USER/envcache"
mkdir -p "$ENVCACHE_ROOT"

sync_tarball_cache() {
  local source_tarball="$1"
  local cache_tarball="$2"
  local expected_sha="$3"
  local lock_path="$4"
  local label="$5"
  mkdir -p "$(dirname "$cache_tarball")"
  exec 7>"$lock_path"
  flock 7
  local cache_sha
  cache_sha="$(sha256sum "$cache_tarball" 2>/dev/null | awk '{print $1}' || true)"
  if [ "$expected_sha" != "$cache_sha" ]; then
    local tmp_tarball="${cache_tarball}.tmp.${SLURM_JOB_ID:-manual}.$$"
    rsync -a --human-readable --info=progress2 --stats "$source_tarball" "$tmp_tarball"
    cache_sha="$(sha256sum "$tmp_tarball" | awk '{print $1}')"
    if [ "$expected_sha" != "$cache_sha" ]; then
      echo "[blocker] tarball_cache_sha_mismatch label=$label expected=$expected_sha observed=$cache_sha" >&2
      exit 38
    fi
    mv "$tmp_tarball" "$cache_tarball"
  fi
  flock -u 7
  exec 7>&-
}

ENV_SOURCE_SHA256="$(sha256sum "/data/topovlm/envs/topovlm_env.tar.gz" | awk '{print $1}')"
ENV_TARBALL_CACHE="$ENVCACHE_ROOT/$ENV_SOURCE_SHA256.tar.gz"
ENV_TARBALL_LOCK="$ENVCACHE_ROOT/$ENV_SOURCE_SHA256.tarball.lock"
sync_tarball_cache "/data/topovlm/envs/topovlm_env.tar.gz" "$ENV_TARBALL_CACHE" "$ENV_SOURCE_SHA256" "$ENV_TARBALL_LOCK" "env"
ENV_DIR="$ENVCACHE_ROOT/$ENV_SOURCE_SHA256"
ENV_LOCK="$ENVCACHE_ROOT/$ENV_SOURCE_SHA256.env.lock"
exec 8>"$ENV_LOCK"
flock 8
if [ -f "$ENV_DIR/.complete" ] && [ -x "$ENV_DIR/bin/python" ]; then
  echo "[cache] env ready path=$ENV_DIR"
else
  if [ -e "$ENV_DIR" ] || [ -L "$ENV_DIR" ]; then
    BROKEN_ENV_DIR="${ENV_DIR}.broken.$(date -u +"%Y%m%dT%H%M%SZ").${SLURM_JOB_ID:-manual}.$$"
    mv "$ENV_DIR" "$BROKEN_ENV_DIR"
    echo "[cache] quarantined incomplete env cache path=$BROKEN_ENV_DIR"
  fi
  mkdir -p "$ENV_DIR"
  tar -xzf "$ENV_TARBALL_CACHE" -C "$ENV_DIR"
  export PATH="$ENV_DIR/bin:$PATH"
  if [ -f "$ENV_DIR/bin/conda-unpack" ]; then
    "$ENV_DIR/bin/conda-unpack"
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$ENV_DIR/.complete"
fi
flock -u 8
exec 8>&-
export PATH="$ENV_DIR/bin:$PATH"
export CPATH="$ENV_DIR/include:${CPATH:-}"
export C_INCLUDE_PATH="$ENV_DIR/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$ENV_DIR/include:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$ENV_DIR/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
write_stage_progress "env" "done" "tarball_env"

stage_readonly_link() {
  local stage_source="$1"
  local cache_target="$2"
  local work_target="$3"
  if [[ "$stage_source" != /* ]]; then
    stage_source="$SUBMIT_DIR/$stage_source"
  fi
  mkdir -p "$(dirname "$cache_target")" "$(dirname "$work_target")"
  if [ -d "$stage_source" ]; then
    mkdir -p "$cache_target"
    rsync -a --human-readable --info=progress2 --stats --update "$stage_source/" "$cache_target/"
  elif [ -f "$stage_source" ]; then
    rsync -a --human-readable --info=progress2 --stats --update "$stage_source" "$cache_target"
  else
    echo "[blocker] stage_input_missing source=$stage_source target=$work_target" >&2
    exit 36
  fi
  if [ -e "$work_target" ] || [ -L "$work_target" ]; then
    echo "[blocker] stage_target_conflict target=$work_target" >&2
    exit 37
  fi
  ln -s "$cache_target" "$work_target"
}
stage_readonly_data() {
  local stage_source="$1"
  local stage_target="$2"
  stage_readonly_link "$stage_source" "$DATA_CACHE/$stage_target" "$WORKDIR/data/$stage_target"
}
stage_readonly_artifact() {
  local stage_source="$1"
  local stage_target="$2"
  stage_readonly_link "$stage_source" "$DATA_CACHE/$stage_target" "$WORKDIR/$stage_target"
}

write_stage_progress "stage_in" "running" "immutable_code"
CODE_LOCK="$CODE_ROOT/$SOURCE_COMMIT.lock"
exec 9>"$CODE_LOCK"
flock 9
if [ -d "$CODE_SOURCE" ] && [ ! -f "$CODE_SOURCE/.complete" ]; then
  echo "[blocker] code_cache_incomplete path=$CODE_SOURCE" >&2
  exit 35
fi
if [ ! -f "$CODE_SOURCE/.complete" ]; then
  mkdir -p "$CODE_SOURCE"
  git -C "$SUBMIT_DIR" archive "$SOURCE_COMMIT" | tar -x -C "$CODE_SOURCE"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$CODE_SOURCE/.complete"
fi
flock -u 9
exec 9>&-
rsync -a --human-readable --info=progress2 --stats --delete "$CODE_SOURCE/" "$WORKDIR/"
write_stage_progress "stage_in" "done" "immutable_code"

write_stage_progress "stage_in" "running" "readonly_inputs"
stage_readonly_data "/data/topovlm/habitat/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2" "topovlm/habitat/datasets/objectnav/hm3d/v2/objectnav_hm3d_v2"
stage_readonly_data "/data/topovlm/habitat/scene_datasets/hm3d" "topovlm/habitat/scene_datasets/hm3d"
stage_readonly_data "/data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b" "topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"
write_stage_progress "stage_in" "done" "readonly_inputs"

cd "$WORKDIR"
if [ "$PWD" != "$WORKDIR" ]; then
  echo "[blocker] workload did not enter scratch workdir" >&2
  exit 21
fi

export RUN_ROOT="$RUN_ROOT"
export OUTPUT_DIR="$RUN_OUTPUT_DIR"
export TMPDIR="$RUN_TMP_DIR"
export XDG_CACHE_HOME="$RUN_TMP_DIR/xdg"
export RUN_LOCAL_DIR="$RUN_LOCAL_DIR"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export CHECKPOINT_DIR="$RUN_OUTPUT_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR" "$XDG_CACHE_HOME"

CHECKPOINT_SYNC_DEST="/home/heejun/Projects/TopoVLM/artifacts/logs/slurm_outputs/topovlm_habitat_h200_smoke/"${SLURM_JOB_ID}"/checkpoints"
export CHECKPOINT_SYNC_DEST
mkdir -p "$CHECKPOINT_SYNC_DEST"
checkpoint_sync_loop() {
  while true; do
    sleep 300
    rsync -a --human-readable --info=progress2 --stats --update "$CHECKPOINT_DIR/" "$CHECKPOINT_SYNC_DEST/"
  done
}
checkpoint_sync_loop &
CHECKPOINT_SYNC_PID="$!"

final_stage_out() {
  local exit_code="$?"
  set +e
  if [ -n "${CHECKPOINT_SYNC_PID:-}" ]; then
    kill "$CHECKPOINT_SYNC_PID" 2>/dev/null || true
    wait "$CHECKPOINT_SYNC_PID" 2>/dev/null || true
  fi
  local dest="/home/heejun/Projects/TopoVLM/artifacts/logs/slurm_outputs/topovlm_habitat_h200_smoke/"${SLURM_JOB_ID}
  mkdir -p "$dest"
  write_stage_progress "stage_out" "running" "$dest"
  rsync -a --human-readable --info=progress2 --stats --update "$RUN_OUTPUT_DIR/" "$dest/"
  write_stage_progress "stage_out" "done" "$dest"
  rsync -a --human-readable --info=progress2 --stats --update "$STAGE_PROGRESS_FILE" "$dest/"
  echo "[stage-out] outputs=$dest final_exit=$exit_code"
  exit "$exit_code"
}
trap final_stage_out EXIT

write_stage_progress "run" "running" "train_command"
stdbuf -oL -eL 'python' 'train.py' '--exp' 'habitat/prismatic_bc_smoke_staged' '--mode' 'preflight' '--allow-missing-data'
write_stage_progress "run" "done" "train_command"

