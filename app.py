from flask import Flask, request, jsonify
import hashlib
import json
import math
import os
import re

app = Flask(__name__)

REQUIRED_FILES = [
    "README.md",
    "training_manifest.json",
    "evaluation.json",
    "inventory.json",
    "adapter_model.safetensors",
    "adapter_config.json",
]

UNSAFE_EXTENSIONS = (
    ".bin",
    ".pt",
    ".pth",
    ".pkl",
    ".pickle",
)

MAX_SAFE_INTEGER = 9007199254740991

MODEL_CARD_PREFIX = "<!-- tds-model-card "


# ============================================================
# HELPERS
# ============================================================

def utf8_bytes(value):
    return value.encode("utf-8")


def sha256_utf8(value):
    return hashlib.sha256(
        utf8_bytes(value)
    ).hexdigest()


def compact_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def add(violations, code):
    violations.add(code)


def nonempty_string(value):
    return (
        isinstance(value, str)
        and len(value) > 0
    )


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


# ============================================================
# REQUEST VALIDATION
# ============================================================

def validate_policy(policy, violations):

    if not isinstance(policy, dict):
        add(violations, "INVALID_POLICY")
        return

    required = policy.get("requiredSlices")

    if (
        not isinstance(required, list)
        or len(required) == 0
        or any(
            not nonempty_string(x)
            for x in required
        )
        or len(set(required)) != len(required)
    ):
        add(violations, "INVALID_POLICY")

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):
        if not nonempty_string(
            policy.get(field)
        ):
            add(violations, "INVALID_POLICY")


def validate_file_values(files, violations):

    for name, value in files.items():

        if not isinstance(name, str):
            add(violations, "INVALID_POLICY")
            continue

        if not isinstance(value, str):
            add(
                violations,
                f"INVALID_FILE:{name}",
            )


def check_required_files(files, violations):

    for name in REQUIRED_FILES:
        if name not in files:
            add(
                violations,
                f"MISSING_FILE:{name}",
            )


def check_extra_and_unsafe_files(
    files,
    violations,
):

    required_names = set(REQUIRED_FILES)

    for name in files:

        if not isinstance(name, str):
            continue

        # Any file outside the six required files is extra.
        if name not in required_names:
            add(
                violations,
                "UNTRACKED_FILE",
            )

        # Unsafe weight extensions are always rejected.
        lower = name.lower()

        if lower.endswith(
            UNSAFE_EXTENSIONS
        ):
            add(
                violations,
                "UNSAFE_WEIGHTS",
            )


# ============================================================
# JSON
# ============================================================

def parse_json_file(
    files,
    name,
    violations,
):

    if name not in files:
        return None

    value = files[name]

    if not isinstance(value, str):
        # Already reported by validate_file_values.
        return None

    try:
        return json.loads(value)

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):
        add(
            violations,
            f"INVALID_JSON:{name}",
        )
        return None


# ============================================================
# INVENTORY
# ============================================================

def verify_inventory(
    files,
    inventory,
    violations,
):

    if not isinstance(inventory, list):

        add(
            violations,
            "INVENTORY_MISMATCH",
        )

        return ""

    # --------------------------------------------------------
    # Build the expected inventory from the actual files.
    # --------------------------------------------------------

    filenames = [
        name
        for name in files
        if name != "inventory.json"
        and isinstance(name, str)
    ]

    # UTF-8 byte ordering.
    filenames.sort(
        key=lambda name: name.encode("utf-8")
    )

    expected = []

    for name in filenames:

        content = files[name]

        # Every bundle file must be a UTF-8 string.
        if not isinstance(content, str):
            add(
                violations,
                f"INVALID_FILE:{name}",
            )
            continue

        raw = content.encode("utf-8")

        expected.append({
            "name": name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(
                raw
            ).hexdigest(),
        })

    # --------------------------------------------------------
    # Validate the structure of the supplied inventory.
    # --------------------------------------------------------

    for entry in inventory:

        if not isinstance(entry, dict):

            add(
                violations,
                "INVENTORY_MISMATCH",
            )

            continue

        # Exact key order is required.
        if list(entry.keys()) != [
            "name",
            "bytes",
            "sha256",
        ]:
            add(
                violations,
                "INVENTORY_MISMATCH",
            )

    # --------------------------------------------------------
    # Exact comparison.
    #
    # This simultaneously checks:
    # - number of entries
    # - filename
    # - byte count
    # - SHA-256
    # - filename order
    # --------------------------------------------------------

    if inventory != expected:

        add(
            violations,
            "INVENTORY_MISMATCH",
        )

    # --------------------------------------------------------
    # Explicitly detect missing tracked files.
    # --------------------------------------------------------

    supplied_names = set()

    for entry in inventory:

        if isinstance(entry, dict):

            name = entry.get("name")

            if isinstance(name, str):
                supplied_names.add(name)

    for name in filenames:

        if name not in supplied_names:

            add(
                violations,
                "UNTRACKED_FILE",
            )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # inventoryDigest is based on the RECOMPUTED inventory,
    # never on the attacker's inventory.json.
    # --------------------------------------------------------

    return sha256_utf8(
        compact_json(expected)
    )


# ============================================================
# ADAPTER CONFIG
# ============================================================

def verify_adapter_config(
    config,
    violations,
):

    if not isinstance(config, dict):

        add(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

        return

    r = config.get("r")
    targets = config.get(
        "target_modules"
    )

    if not safe_integer(r):

        add(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    if (
        not isinstance(targets, list)
        or len(targets) == 0
        or any(
            not nonempty_string(x)
            for x in targets
        )
        or len(set(targets))
        != len(targets)
    ):

        add(
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

    if not isinstance(manifest, dict):

        add(
            violations,
            "INVALID_TRAINING_MANIFEST",
        )

        return

    fields = [
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ]

    for field in fields:

        if field not in manifest:

            add(
                violations,
                f"MISSING_MANIFEST_FIELD:{field}",
            )

        elif not nonempty_string(
            manifest[field]
        ):

            add(
                violations,
                "INVALID_TRAINING_MANIFEST",
            )

    # --------------------------------------------------------
    # Immutable base revision.
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

        add(
            violations,
            "MUTABLE_BASE_REVISION",
        )

    # --------------------------------------------------------
    # MODEL ARTIFACT DIGEST
    # --------------------------------------------------------

    if (
        "adapter_model.safetensors"
        in files
    ):

        model = files[
            "adapter_model.safetensors"
        ]

        if isinstance(model, str):

            actual = sha256_utf8(
                model
            )

            expected = manifest.get(
                "modelArtifactDigest"
            )

            if actual != expected:

                add(
                    violations,
                    "MODEL_ARTIFACT_MISMATCH",
                )

    # --------------------------------------------------------
    # EVALUATION ARTIFACT DIGEST
    # --------------------------------------------------------

    if "evaluation.json" in files:

        evaluation = files[
            "evaluation.json"
        ]

        if isinstance(
            evaluation,
            str,
        ):

            actual = sha256_utf8(
                evaluation
            )

            expected = manifest.get(
                "evaluationArtifactDigest"
            )

            if actual != expected:

                add(
                    violations,
                    "EVALUATION_ARTIFACT_MISMATCH",
                )


# ============================================================
# EVALUATION
# ============================================================

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

        add(
            violations,
            "INVALID_EVALUATION",
        )

        return

    # Evaluation must bind to this exact model.
    if (
        evaluation.get(
            "modelArtifactDigest"
        )
        != manifest.get(
            "modelArtifactDigest"
        )
    ):

        add(
            violations,
            "EVALUATION_DIGEST_MISMATCH",
        )

    # Aggregate.
    aggregate = evaluation.get(
        "aggregate"
    )

    if (
        not isinstance(
            aggregate,
            (int, float),
        )
        or isinstance(
            aggregate,
            bool,
        )
        or not math.isfinite(
            aggregate
        )
        or aggregate < 0
        or aggregate > 1
    ):

        add(
            violations,
            "INVALID_AGGREGATE",
        )

    # Required slices.
    slices = evaluation.get(
        "slices"
    )

    if not isinstance(
        slices,
        dict,
    ):

        for required in policy.get(
            "requiredSlices",
            [],
        ):

            add(
                violations,
                f"MISSING_SLICE:{required}",
            )

        return

    for required in policy.get(
        "requiredSlices",
        [],
    ):

        if required not in slices:

            add(
                violations,
                f"MISSING_SLICE:{required}",
            )

            continue

        value = slices[
            required
        ]

        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(
                value,
                bool,
            )
            or not math.isfinite(
                value
            )
            or value < 0
            or value > 1
        ):

            add(
                violations,
                f"SLICE_RANGE:{required}",
            )


# ============================================================
# MODEL CARD
# ============================================================

def find_model_cards(readme):

    positions = []

    start = 0

    while True:

        position = readme.find(
            MODEL_CARD_PREFIX,
            start,
        )

        if position == -1:
            break

        positions.append(
            position
        )

        start = (
            position
            + len(MODEL_CARD_PREFIX)
        )

    return positions


def verify_model_card(
    readme,
    manifest,
    policy,
    violations,
):

    positions = find_model_cards(
        readme
    )

    # --------------------------------------------------------
    # ZERO MARKERS
    # --------------------------------------------------------

    if len(positions) == 0:

        add(
            violations,
            "MODEL_CARD_COUNT",
        )

        add(
            violations,
            "MISSING_MODEL_CARD",
        )

        return

    # --------------------------------------------------------
    # MULTIPLE MARKERS
    # --------------------------------------------------------

    if len(positions) > 1:

        add(
            violations,
            "MODEL_CARD_COUNT",
        )

        return

    # --------------------------------------------------------
    # ONE MARKER
    # --------------------------------------------------------

    start = (
        positions[0]
        + len(MODEL_CARD_PREFIX)
    )

    end = readme.find(
        "-->",
        start,
    )

    if end == -1:

        add(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    payload = readme[
        start:end
    ].strip()

    # json.loads correctly understands braces inside strings.
    try:

        card = json.loads(
            payload
        )

    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
    ):

        add(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    if not isinstance(
        card,
        dict,
    ):

        add(
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

            add(
                violations,
                "MODEL_CARD_MISMATCH",
            )

            return


# ============================================================
# ENDPOINT
# ============================================================

@app.post("/verify-bundle")
def verify_bundle():

    # --------------------------------------------------------
    # HTTP input validation
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

    # Missing policy = HTTP 400.
    if "policy" not in body:

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # files must be an object.
    files = body.get(
        "files"
    )

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
    # Basic checks
    # --------------------------------------------------------

    validate_policy(
        policy,
        violations,
    )

    validate_file_values(
        files,
        violations,
    )

    check_required_files(
        files,
        violations,
    )

    check_extra_and_unsafe_files(
        files,
        violations,
    )

    # --------------------------------------------------------
    # Parse JSON files
    # --------------------------------------------------------

    inventory = parse_json_file(
        files,
        "inventory.json",
        violations,
    )

    adapter_config = parse_json_file(
        files,
        "adapter_config.json",
        violations,
    )

    manifest = parse_json_file(
        files,
        "training_manifest.json",
        violations,
    )

    evaluation = parse_json_file(
        files,
        "evaluation.json",
        violations,
    )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    inventory_digest = ""

    if "inventory.json" in files:

        inventory_digest = (
            verify_inventory(
                files,
                inventory,
                violations,
            )
        )

    # --------------------------------------------------------
    # Adapter config
    # --------------------------------------------------------

    if "adapter_config.json" in files:

        verify_adapter_config(
            adapter_config,
            violations,
        )

    # --------------------------------------------------------
    # Training manifest
    # --------------------------------------------------------

    if "training_manifest.json" in files:

        verify_training_manifest(
            manifest,
            files,
            violations,
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        "evaluation.json" in files
        and isinstance(
            manifest,
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
    # Model card
    # --------------------------------------------------------

    if "README.md" in files:

        readme = files[
            "README.md"
        ]

        if isinstance(
            readme,
            str,
        ):

            if isinstance(
                manifest,
                dict,
            ):

                verify_model_card(
                    readme,
                    manifest,
                    policy,
                    violations,
                )

            else:

                positions = find_model_cards(
                    readme
                )

                if len(positions) == 0:

                    add(
                        violations,
                        "MODEL_CARD_COUNT",
                    )

                    add(
                        violations,
                        "MISSING_MODEL_CARD",
                    )

                elif len(positions) > 1:

                    add(
                        violations,
                        "MODEL_CARD_COUNT",
                    )

                else:

                    add(
                        violations,
                        "MODEL_CARD_MISMATCH",
                    )

    # --------------------------------------------------------
    # Deterministic output
    # --------------------------------------------------------

    sorted_violations = sorted(
        violations,
        key=lambda value:
            value.encode("utf-8"),
    )

    decision = (
        "admit"
        if len(sorted_violations) == 0
        else "reject"
    )

    return jsonify({
        "decision": decision,
        "violations": sorted_violations,
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
