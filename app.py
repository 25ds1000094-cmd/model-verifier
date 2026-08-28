from flask import Flask, request, jsonify
import hashlib
import json
import math
import os
import re

app = Flask(__name__)

REQUIRED_FILES = (
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
)

UNSAFE_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
)

SAFE_INTEGER_MAX = 9007199254740991

MODEL_CARD_PREFIX = '<!-- tds-model-card '


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_nonempty_string(value):
    return isinstance(value, str) and len(value) > 0


def is_safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 1
        and value <= SAFE_INTEGER_MAX
    )


def utf8(value):
    return value.encode("utf-8")


def sha256(value):
    return hashlib.sha256(
        utf8(value)
    ).hexdigest()


def compact_json(value):
    """
    Compact JSON representation.

    ensure_ascii=False means Unicode characters are represented
    as their actual UTF-8 characters rather than \\u escapes.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def violation(violations, code):
    violations.add(code)


# ============================================================
# POLICY
# ============================================================

def validate_policy(policy, violations):

    if not isinstance(policy, dict):
        violation(
            violations,
            "INVALID_POLICY",
        )
        return

    required_slices = policy.get(
        "requiredSlices"
    )

    if not isinstance(
        required_slices,
        list,
    ):
        violation(
            violations,
            "INVALID_POLICY",
        )

    else:

        if len(required_slices) == 0:
            violation(
                violations,
                "INVALID_POLICY",
            )

        if any(
            not is_nonempty_string(x)
            for x in required_slices
        ):
            violation(
                violations,
                "INVALID_POLICY",
            )

        if len(set(required_slices)) != len(
            required_slices
        ):
            violation(
                violations,
                "INVALID_POLICY",
            )

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):

        if not is_nonempty_string(
            policy.get(field)
        ):
            violation(
                violations,
                "INVALID_POLICY",
            )


# ============================================================
# FILE INPUT
# ============================================================

def validate_files(files, violations):

    for name, content in files.items():

        if not isinstance(name, str):
            violation(
                violations,
                "INVALID_POLICY",
            )
            continue

        # The request says files are UTF-8 strings.
        if not isinstance(content, str):
            violation(
                violations,
                f"INVALID_FILE:{name}",
            )


def check_required_files(
    files,
    violations,
):

    for name in REQUIRED_FILES:

        if name not in files:
            violation(
                violations,
                f"MISSING_FILE:{name}",
            )


def check_extra_files(
    files,
    violations,
):

    required = set(REQUIRED_FILES)

    for name in files:

        if not isinstance(name, str):
            continue

        if name not in required:
            violation(
                violations,
                "UNTRACKED_FILE",
            )


def check_unsafe_extensions(
    files,
    violations,
):

    for name in files:

        if not isinstance(name, str):
            continue

        lowered = name.lower()

        if lowered.endswith(
            UNSAFE_EXTENSIONS
        ):
            violation(
                violations,
                "UNSAFE_WEIGHTS",
            )


# ============================================================
# JSON PARSING
# ============================================================

def parse_json(
    files,
    filename,
    violations,
):

    if filename not in files:
        return None

    content = files[filename]

    if not isinstance(content, str):
        return None

    try:
        return json.loads(content)

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        violation(
            violations,
            f"INVALID_JSON:{filename}",
        )

        return None


# ============================================================
# INVENTORY
# ============================================================

def recompute_inventory(
    files,
    violations,
):
    """
    Construct the inventory from the actual submitted files.

    inventory.json itself is excluded.

    Every other file must be a UTF-8 string.
    """

    names = []

    for name in files:

        if name == "inventory.json":
            continue

        if isinstance(name, str):
            names.append(name)

    # Sort by UTF-8 bytes, not locale or case.
    names.sort(
        key=lambda name: name.encode("utf-8")
    )

    result = []

    for name in names:

        content = files[name]

        if not isinstance(content, str):
            violation(
                violations,
                f"INVALID_FILE:{name}",
            )
            continue

        raw = content.encode("utf-8")

        result.append({
            "name": name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(
                raw
            ).hexdigest(),
        })

    return result


def verify_inventory(
    files,
    supplied_inventory,
    violations,
):

    # If inventory isn't a JSON array,
    # it cannot be a valid inventory.
    if not isinstance(
        supplied_inventory,
        list,
    ):

        violation(
            violations,
            "INVENTORY_MISMATCH",
        )

        # Still calculate a digest from the
        # actual files.
        expected = recompute_inventory(
            files,
            violations,
        )

        return sha256(
            compact_json(expected)
        )

    expected = recompute_inventory(
        files,
        violations,
    )

    # --------------------------------------------------------
    # Each inventory entry must have EXACTLY:
    #
    # name, bytes, sha256
    #
    # and that exact key order.
    # --------------------------------------------------------

    structure_valid = True

    for entry in supplied_inventory:

        if not isinstance(
            entry,
            dict,
        ):

            structure_valid = False
            break

        if list(entry.keys()) != [
            "name",
            "bytes",
            "sha256",
        ]:

            structure_valid = False
            break

    if not structure_valid:

        violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    # --------------------------------------------------------
    # Exact list comparison.
    #
    # This checks:
    # - number of entries
    # - names
    # - byte counts
    # - hashes
    # - order
    # - duplicates
    # - extra inventory entries
    # --------------------------------------------------------

    if supplied_inventory != expected:

        violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    # --------------------------------------------------------
    # Explicitly detect files missing from inventory.
    # --------------------------------------------------------

    tracked_names = set()

    for entry in supplied_inventory:

        if isinstance(entry, dict):

            name = entry.get("name")

            if isinstance(name, str):
                tracked_names.add(name)

    for entry in expected:

        name = entry["name"]

        if name not in tracked_names:

            violation(
                violations,
                "UNTRACKED_FILE",
            )

    # --------------------------------------------------------
    # inventoryDigest MUST be calculated from the
    # RECOMPUTED inventory.
    # --------------------------------------------------------

    return sha256(
        compact_json(expected)
    )


# ============================================================
# ADAPTER CONFIG
# ============================================================

def verify_adapter_config(
    config,
    violations,
):

    if not isinstance(
        config,
        dict,
    ):

        violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

        return

    r = config.get("r")

    targets = config.get(
        "target_modules"
    )

    if not is_safe_integer(r):

        violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    if not isinstance(
        targets,
        list,
    ):

        violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    else:

        if len(targets) == 0:

            violation(
                violations,
                "INVALID_ADAPTER_CONFIG",
            )

        if any(
            not is_nonempty_string(x)
            for x in targets
        ):

            violation(
                violations,
                "INVALID_ADAPTER_CONFIG",
            )

        if len(set(targets)) != len(
            targets
        ):

            violation(
                violations,
                "INVALID_ADAPTER_CONFIG",
            )


# ============================================================
# TRAINING MANIFEST
# ============================================================

def verify_training_manifest(
    manifest,
    files,
    violations,
):

    if not isinstance(
        manifest,
        dict,
    ):

        violation(
            violations,
            "INVALID_TRAINING_MANIFEST",
        )

        return

    required_fields = (
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    )

    # --------------------------------------------------------
    # Required fields.
    # --------------------------------------------------------

    for field in required_fields:

        if field not in manifest:

            violation(
                violations,
                f"MISSING_MANIFEST_FIELD:{field}",
            )

        elif not is_nonempty_string(
            manifest[field]
        ):

            violation(
                violations,
                "INVALID_TRAINING_MANIFEST",
            )

    # --------------------------------------------------------
    # baseRevision must be exactly:
    #
    # 40 lowercase hexadecimal characters.
    # --------------------------------------------------------

    revision = manifest.get(
        "baseRevision"
    )

    if (
        not isinstance(revision, str)
        or re.fullmatch(
            r"[0-9a-f]{40}",
            revision,
        ) is None
    ):

        violation(
            violations,
            "MUTABLE_BASE_REVISION",
        )

    # --------------------------------------------------------
    # Model artifact digest.
    # --------------------------------------------------------

    if (
        "adapter_model.safetensors"
        in files
    ):

        model = files[
            "adapter_model.safetensors"
        ]

        if isinstance(
            model,
            str,
        ):

            actual_model_digest = sha256(
                model
            )

            declared_model_digest = (
                manifest.get(
                    "modelArtifactDigest"
                )
            )

            if (
                actual_model_digest
                != declared_model_digest
            ):

                violation(
                    violations,
                    "MODEL_ARTIFACT_MISMATCH",
                )

    # --------------------------------------------------------
    # Evaluation artifact digest.
    # --------------------------------------------------------

    if "evaluation.json" in files:

        evaluation = files[
            "evaluation.json"
        ]

        if isinstance(
            evaluation,
            str,
        ):

            actual_evaluation_digest = (
                sha256(evaluation)
            )

            declared_evaluation_digest = (
                manifest.get(
                    "evaluationArtifactDigest"
                )
            )

            if (
                actual_evaluation_digest
                != declared_evaluation_digest
            ):

                violation(
                    violations,
                    "EVALUATION_ARTIFACT_MISMATCH",
                )


# ============================================================
# EVALUATION
# ============================================================

def valid_score(value):

    return (
        isinstance(
            value,
            (int, float),
        )
        and not isinstance(
            value,
            bool,
        )
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def verify_evaluation(
    evaluation,
    manifest,
    policy,
    violations,
):

    if not isinstance(
        evaluation,
        dict,
    ):

        violation(
            violations,
            "INVALID_EVALUATION",
        )

        return

    # --------------------------------------------------------
    # Bind evaluation to the exact model artifact.
    # --------------------------------------------------------

    if (
        evaluation.get(
            "modelArtifactDigest"
        )
        != manifest.get(
            "modelArtifactDigest"
        )
    ):

        violation(
            violations,
            "EVALUATION_DIGEST_MISMATCH",
        )

    # --------------------------------------------------------
    # Aggregate.
    # --------------------------------------------------------

    if not valid_score(
        evaluation.get("aggregate")
    ):

        violation(
            violations,
            "INVALID_AGGREGATE",
        )

    # --------------------------------------------------------
    # Required slices.
    # --------------------------------------------------------

    slices = evaluation.get(
        "slices"
    )

    if not isinstance(
        slices,
        dict,
    ):

        for name in policy.get(
            "requiredSlices",
            [],
        ):

            violation(
                violations,
                f"MISSING_SLICE:{name}",
            )

        return

    for name in policy.get(
        "requiredSlices",
        [],
    ):

        if name not in slices:

            violation(
                violations,
                f"MISSING_SLICE:{name}",
            )

            continue

        if not valid_score(
            slices[name]
        ):

            violation(
                violations,
                f"SLICE_RANGE:{name}",
            )


# ============================================================
# MODEL CARD
# ============================================================

def find_model_card_markers(
    readme,
):

    result = []

    position = 0

    while True:

        found = readme.find(
            MODEL_CARD_PREFIX,
            position,
        )

        if found == -1:
            break

        result.append(found)

        position = (
            found
            + len(MODEL_CARD_PREFIX)
        )

    return result


def verify_model_card(
    readme,
    manifest,
    policy,
    violations,
):

    markers = find_model_card_markers(
        readme
    )

    # --------------------------------------------------------
    # No marker:
    #
    # MODEL_CARD_COUNT + MISSING_MODEL_CARD
    # --------------------------------------------------------

    if len(markers) == 0:

        violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        violation(
            violations,
            "MISSING_MODEL_CARD",
        )

        return

    # --------------------------------------------------------
    # Multiple markers:
    #
    # ONLY MODEL_CARD_COUNT
    # --------------------------------------------------------

    if len(markers) > 1:

        violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        return

    # --------------------------------------------------------
    # Exactly one marker.
    # --------------------------------------------------------

    start = (
        markers[0]
        + len(MODEL_CARD_PREFIX)
    )

    end = readme.find(
        "-->",
        start,
    )

    if end == -1:

        violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    payload = readme[
        start:end
    ]

    # The entire payload between the prefix and -->
    # is parsed. Do not attempt to parse braces manually.
    try:

        card = json.loads(
            payload
        )

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    # Parsed JSON must be an object.
    if not isinstance(
        card,
        dict,
    ):

        violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    expected = {
        "task": manifest.get(
            "task"
        ),
        "baseRevision": manifest.get(
            "baseRevision"
        ),
        "datasetDigest": manifest.get(
            "datasetDigest"
        ),
        "modelArtifactDigest": manifest.get(
            "modelArtifactDigest"
        ),
        "license": policy.get(
            "license"
        ),
        "intendedUse": policy.get(
            "intendedUse"
        ),
        "limitations": policy.get(
            "limitations"
        ),
    }

    for field, expected_value in expected.items():

        if card.get(field) != expected_value:

            violation(
                violations,
                "MODEL_CARD_MISMATCH",
            )

            # Only one code is needed.
            return


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/verify-bundle")
def verify_bundle():

    # --------------------------------------------------------
    # HTTP input validation.
    # --------------------------------------------------------

    if not request.is_json:

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    body = request.get_json(
        silent=True
    )

    if not isinstance(
        body,
        dict,
    ):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # A missing policy is specifically HTTP 400.
    if "policy" not in body:

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    files = body.get(
        "files"
    )

    # A missing files property or non-object files
    # is invalid input.
    if not isinstance(
        files,
        dict,
    ):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    policy = body.get(
        "policy"
    )

    violations = set()

    # --------------------------------------------------------
    # Basic validation.
    # --------------------------------------------------------

    validate_policy(
        policy,
        violations,
    )

    validate_files(
        files,
        violations,
    )

    check_required_files(
        files,
        violations,
    )

    check_extra_files(
        files,
        violations,
    )

    check_unsafe_extensions(
        files,
        violations,
    )

    # --------------------------------------------------------
    # Parse JSON documents.
    # --------------------------------------------------------

    inventory = parse_json(
        files,
        "inventory.json",
        violations,
    )

    adapter_config = parse_json(
        files,
        "adapter_config.json",
        violations,
    )

    manifest = parse_json(
        files,
        "training_manifest.json",
        violations,
    )

    evaluation = parse_json(
        files,
        "evaluation.json",
        violations,
    )

    # --------------------------------------------------------
    # Inventory verification.
    # --------------------------------------------------------

    if "inventory.json" in files:

        inventory_digest = verify_inventory(
            files,
            inventory,
            violations,
        )

    else:

        # No inventory file exists, so there is no
        # supplied inventory to trust. Calculate the
        # digest from what can be recomputed anyway.
        expected_inventory = recompute_inventory(
            files,
            violations,
        )

        inventory_digest = sha256(
            compact_json(
                expected_inventory
            )
        )

    # --------------------------------------------------------
    # Adapter configuration.
    # --------------------------------------------------------

    if "adapter_config.json" in files:

        verify_adapter_config(
            adapter_config,
            violations,
        )

    # --------------------------------------------------------
    # Training manifest.
    # --------------------------------------------------------

    if "training_manifest.json" in files:

        verify_training_manifest(
            manifest,
            files,
            violations,
        )

    # --------------------------------------------------------
    # Evaluation.
    # --------------------------------------------------------

    if (
        "evaluation.json" in files
        and isinstance(
            manifest,
            dict,
        )
        and isinstance(
            policy,
            dict,
        )
    ):

        verify_evaluation(
            evaluation,
            manifest,
            policy,
            violations,
        )

    # --------------------------------------------------------
    # Model card.
    # --------------------------------------------------------

    if "README.md" in files:

        readme = files[
            "README.md"
        ]

        if isinstance(
            readme,
            str,
        ):

            markers = (
                find_model_card_markers(
                    readme
                )
            )

            # If multiple markers exist, the assignment says
            # ONLY MODEL_CARD_COUNT.
            if len(markers) > 1:

                violation(
                    violations,
                    "MODEL_CARD_COUNT",
                )

            elif len(markers) == 0:

                violation(
                    violations,
                    "MODEL_CARD_COUNT",
                )

                violation(
                    violations,
                    "MISSING_MODEL_CARD",
                )

            elif isinstance(
                manifest,
                dict,
            ) and isinstance(
                policy,
                dict,
            ):

                verify_model_card(
                    readme,
                    manifest,
                    policy,
                    violations,
                )

            else:

                # There is one marker but the machine-readable
                # information needed for comparison is invalid.
                # The marker itself is present, so do not report
                # it as missing.
                violation(
                    violations,
                    "MODEL_CARD_MISMATCH",
                )

    # --------------------------------------------------------
    # Deterministic violation ordering.
    # --------------------------------------------------------

    ordered_violations = sorted(
        violations,
        key=lambda x: x.encode("utf-8"),
    )

    decision = (
        "admit"
        if len(ordered_violations) == 0
        else "reject"
    )

    # EXACT response shape.
    return jsonify({
        "decision": decision,
        "violations": ordered_violations,
        "inventoryDigest": inventory_digest,
    })


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
