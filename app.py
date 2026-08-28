from flask import Flask, request, jsonify
import hashlib
import json
import math
import os
import re


app = Flask(__name__)


# ============================================================
# CONSTANTS
# ============================================================

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
# BASIC HELPERS
# ============================================================

def utf8_bytes(text):
    """Convert a Python string to its exact UTF-8 bytes."""
    return text.encode("utf-8")


def sha256_text(text):
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256(utf8_bytes(text)).hexdigest()


def compact_json(value):
    """
    Produce compact JSON.

    This is important for inventoryDigest because the assignment
    requires the exact compact JSON representation.
    """
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def add_violation(violations, code):
    """Use a set so violations are automatically deduplicated."""
    violations.add(code)


def non_empty_string(value):
    return isinstance(value, str) and len(value) > 0


def safe_integer(value):
    """
    JavaScript-style safe positive integer.
    """
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 1 <= value <= MAX_SAFE_INTEGER
    )


# ============================================================
# JSON FILE HANDLING
# ============================================================

def parse_json_file(files, filename, violations):
    """
    Safely parse one JSON file.

    Returns None when:
    - the file doesn't exist
    - the file isn't a string
    - the JSON is invalid
    """

    if filename not in files:
        return None

    content = files[filename]

    if not isinstance(content, str):
        add_violation(
            violations,
            f"INVALID_FILE:{filename}",
        )
        return None

    try:
        return json.loads(content)

    except (
        json.JSONDecodeError,
        UnicodeEncodeError,
        TypeError,
        ValueError,
    ):
        add_violation(
            violations,
            f"INVALID_JSON:{filename}",
        )
        return None


# ============================================================
# POLICY VALIDATION
# ============================================================

def validate_policy(policy, violations):
    """
    Validate:

    requiredSlices:
      - non-empty array
      - unique
      - every value is a non-empty string

    license:
      - non-empty string

    intendedUse:
      - non-empty string

    limitations:
      - non-empty string
    """

    if not isinstance(policy, dict):
        add_violation(
            violations,
            "INVALID_POLICY",
        )
        return

    required_slices = policy.get("requiredSlices")

    if (
        not isinstance(required_slices, list)
        or len(required_slices) == 0
        or any(
            not non_empty_string(x)
            for x in required_slices
        )
        or len(set(required_slices)) != len(required_slices)
    ):
        add_violation(
            violations,
            "INVALID_POLICY",
        )

    for field in (
        "license",
        "intendedUse",
        "limitations",
    ):
        if not non_empty_string(policy.get(field)):
            add_violation(
                violations,
                "INVALID_POLICY",
            )


# ============================================================
# FILE VALIDATION
# ============================================================

def validate_files(files, violations):
    """
    Validate the basic structure of files.

    Invalid file contents are reported as INVALID_FILE:<name>
    instead of crashing the verifier.
    """

    if not isinstance(files, dict):
        return

    for filename, content in files.items():

        if not isinstance(filename, str):
            add_violation(
                violations,
                "INVALID_POLICY",
            )
            continue

        if not isinstance(content, str):
            add_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )


def check_required_files(files, violations):
    for filename in REQUIRED_FILES:
        if filename not in files:
            add_violation(
                violations,
                f"MISSING_FILE:{filename}",
            )


def check_unsafe_weights(files, violations):
    """
    .bin, .pt, .pth, .pkl and .pickle are unsafe.
    """

    for filename in files:

        if not isinstance(filename, str):
            continue

        if filename.lower().endswith(
            UNSAFE_EXTENSIONS
        ):
            add_violation(
                violations,
                "UNSAFE_WEIGHTS",
            )


# ============================================================
# INVENTORY
# ============================================================

def verify_inventory(files, inventory, violations):
    """
    inventory.json must contain every file except itself.

    Each entry must have exactly these keys, in this order:

        name
        bytes
        sha256

    Files must be sorted by UTF-8 filename bytes.

    The inventoryDigest is calculated from the recomputed
    inventory, NOT from the attacker's supplied inventory.
    """

    if not isinstance(inventory, list):
        add_violation(
            violations,
            "INVENTORY_MISMATCH",
        )
        return ""

    actual_names = [
        filename
        for filename in files
        if filename != "inventory.json"
    ]

    # Required UTF-8 byte ordering.
    actual_names.sort(
        key=lambda x: x.encode("utf-8")
        if isinstance(x, str)
        else b""
    )

    expected_inventory = []

    for filename in actual_names:

        content = files[filename]

        # Prevent malicious input from crashing the server.
        if not isinstance(content, str):
            add_violation(
                violations,
                f"INVALID_FILE:{filename}",
            )
            continue

        raw = utf8_bytes(content)

        expected_inventory.append({
            "name": filename,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })

    # --------------------------------------------------------
    # Check that every supplied inventory entry is an object.
    # --------------------------------------------------------

    for entry in inventory:

        if not isinstance(entry, dict):
            add_violation(
                violations,
                "INVENTORY_MISMATCH",
            )
            continue

        # Exact key order.
        if list(entry.keys()) != [
            "name",
            "bytes",
            "sha256",
        ]:
            add_violation(
                violations,
                "INVENTORY_MISMATCH",
            )

    # --------------------------------------------------------
    # Check exact values and ordering.
    # --------------------------------------------------------

    if inventory != expected_inventory:
        add_violation(
            violations,
            "INVENTORY_MISMATCH",
        )

    # --------------------------------------------------------
    # Check for files omitted from inventory.
    # --------------------------------------------------------

    inventory_names = set()

    for entry in inventory:

        if isinstance(entry, dict):

            name = entry.get("name")

            if isinstance(name, str):
                inventory_names.add(name)

    for filename in actual_names:

        if filename not in inventory_names:
            add_violation(
                violations,
                "UNTRACKED_FILE",
            )

    # --------------------------------------------------------
    # Calculate inventoryDigest from the recomputed inventory.
    # --------------------------------------------------------

    return sha256_text(
        compact_json(expected_inventory)
    )


# ============================================================
# ADAPTER CONFIG
# ============================================================

def verify_adapter_config(config, violations):
    """
    Required:

        {
          "r": positive safe integer,
          "target_modules": [
             "module1",
             "module2"
          ]
        }

    Extra properties are allowed.
    """

    if not isinstance(config, dict):
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )
        return

    r = config.get("r")
    target_modules = config.get(
        "target_modules"
    )

    if not safe_integer(r):
        add_violation(
            violations,
            "INVALID_ADAPTER_CONFIG",
        )

    if (
        not isinstance(target_modules, list)
        or len(target_modules) == 0
        or any(
            not non_empty_string(x)
            for x in target_modules
        )
        or len(set(target_modules))
        != len(target_modules)
    ):
        add_violation(
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
    """
    Verify training manifest structure and artifact digests.
    """

    if not isinstance(manifest, dict):
        add_violation(
            violations,
            "INVALID_TRAINING_MANIFEST",
        )
        return

    required_fields = [
        "baseRevision",
        "task",
        "datasetDigest",
        "codeDigest",
        "trainingConfigDigest",
        "modelArtifactDigest",
        "evaluationArtifactDigest",
    ]

    # --------------------------------------------------------
    # Required fields.
    # --------------------------------------------------------

    for field in required_fields:

        if field not in manifest:

            add_violation(
                violations,
                f"MISSING_MANIFEST_FIELD:{field}",
            )

        elif not non_empty_string(
            manifest[field]
        ):

            add_violation(
                violations,
                "INVALID_TRAINING_MANIFEST",
            )

    # --------------------------------------------------------
    # Immutable 40-character lowercase hexadecimal revision.
    # --------------------------------------------------------

    base_revision = manifest.get(
        "baseRevision"
    )

    if (
        not isinstance(base_revision, str)
        or re.fullmatch(
            r"[0-9a-f]{40}",
            base_revision,
        ) is None
    ):
        add_violation(
            violations,
            "MUTABLE_BASE_REVISION",
        )

    # --------------------------------------------------------
    # Model artifact digest.
    # --------------------------------------------------------

    if "adapter_model.safetensors" in files:

        content = files[
            "adapter_model.safetensors"
        ]

        if isinstance(content, str):

            actual_model_digest = sha256_text(
                content
            )

            if (
                manifest.get(
                    "modelArtifactDigest"
                )
                != actual_model_digest
            ):
                add_violation(
                    violations,
                    "MODEL_ARTIFACT_MISMATCH",
                )

    # --------------------------------------------------------
    # Evaluation artifact digest.
    # --------------------------------------------------------

    if "evaluation.json" in files:

        content = files["evaluation.json"]

        if isinstance(content, str):

            actual_evaluation_digest = (
                sha256_text(content)
            )

            if (
                manifest.get(
                    "evaluationArtifactDigest"
                )
                != actual_evaluation_digest
            ):
                add_violation(
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
    """
    Verify:

    - evaluation is an object
    - model digest matches manifest
    - aggregate is finite and in [0,1]
    - every required slice exists
    - every required slice is finite and in [0,1]
    """

    if not isinstance(evaluation, dict):

        add_violation(
            violations,
            "INVALID_EVALUATION",
        )
        return

    # --------------------------------------------------------
    # Bind evaluation to the model.
    # --------------------------------------------------------

    if (
        evaluation.get(
            "modelArtifactDigest"
        )
        != manifest.get(
            "modelArtifactDigest"
        )
    ):
        add_violation(
            violations,
            "EVALUATION_DIGEST_MISMATCH",
        )

    # --------------------------------------------------------
    # Aggregate.
    # --------------------------------------------------------

    aggregate = evaluation.get(
        "aggregate"
    )

    if (
        not isinstance(
            aggregate,
            (int, float),
        )
        or isinstance(aggregate, bool)
        or not math.isfinite(aggregate)
        or aggregate < 0
        or aggregate > 1
    ):
        add_violation(
            violations,
            "INVALID_AGGREGATE",
        )

    # --------------------------------------------------------
    # Required slices.
    # --------------------------------------------------------

    slices = evaluation.get(
        "slices"
    )

    if not isinstance(slices, dict):

        for required in policy.get(
            "requiredSlices",
            [],
        ):
            add_violation(
                violations,
                f"MISSING_SLICE:{required}",
            )

        return

    for required in policy.get(
        "requiredSlices",
        [],
    ):

        if required not in slices:

            add_violation(
                violations,
                f"MISSING_SLICE:{required}",
            )

            continue

        value = slices[required]

        if (
            not isinstance(
                value,
                (int, float),
            )
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            or value > 1
        ):
            add_violation(
                violations,
                f"SLICE_RANGE:{required}",
            )


# ============================================================
# MODEL CARD
# ============================================================

def find_model_cards(readme):
    """
    Find occurrences of the exact model-card prefix.

    We do NOT try to count braces manually.

    This is important because braces inside JSON strings,
    such as "{still text}", are ordinary string characters.
    """

    positions = []

    start = 0

    while True:

        position = readme.find(
            MODEL_CARD_PREFIX,
            start,
        )

        if position == -1:
            break

        positions.append(position)

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
    # Zero markers.
    # --------------------------------------------------------

    if len(positions) == 0:

        add_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        add_violation(
            violations,
            "MISSING_MODEL_CARD",
        )

        return

    # --------------------------------------------------------
    # More than one marker.
    # --------------------------------------------------------

    if len(positions) > 1:

        add_violation(
            violations,
            "MODEL_CARD_COUNT",
        )

        return

    # --------------------------------------------------------
    # Exactly one marker.
    # --------------------------------------------------------

    payload_start = (
        positions[0]
        + len(MODEL_CARD_PREFIX)
    )

    payload_end = readme.find(
        "-->",
        payload_start,
    )

    if payload_end == -1:

        add_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    payload = readme[
        payload_start:payload_end
    ].strip()

    # --------------------------------------------------------
    # Parse JSON.
    # --------------------------------------------------------

    try:

        card = json.loads(
            payload
        )

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):

        add_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    # --------------------------------------------------------
    # Payload must be an object.
    # --------------------------------------------------------

    if not isinstance(card, dict):

        add_violation(
            violations,
            "INVALID_MODEL_CARD",
        )

        return

    # --------------------------------------------------------
    # Required model-card fields.
    # --------------------------------------------------------

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

            add_violation(
                violations,
                "MODEL_CARD_MISMATCH",
            )

            return


# ============================================================
# MAIN ENDPOINT
# ============================================================

@app.post("/verify-bundle")
def verify_bundle():

    # ========================================================
    # 1. Validate HTTP request.
    # ========================================================

    if not request.is_json:

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    body = request.get_json(
        silent=True
    )

    if not isinstance(body, dict):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # Missing policy must produce HTTP 400.
    if "policy" not in body:

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    # files must be an object.
    files = body.get(
        "files"
    )

    if not isinstance(files, dict):

        return jsonify({
            "error": "INVALID_INPUT"
        }), 400

    policy = body.get(
        "policy"
    )

    # ========================================================
    # 2. Create violation set.
    # ========================================================

    violations = set()

    # ========================================================
    # 3. Validate policy/files.
    # ========================================================

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

    check_unsafe_weights(
        files,
        violations,
    )

    # ========================================================
    # 4. Parse JSON files.
    # ========================================================

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

    # ========================================================
    # 5. Verify inventory.
    # ========================================================

    inventory_digest = ""

    if "inventory.json" in files:

        inventory_digest = (
            verify_inventory(
                files,
                inventory,
                violations,
            )
        )

    # ========================================================
    # 6. Verify adapter config.
    # ========================================================

    if "adapter_config.json" in files:

        verify_adapter_config(
            adapter_config,
            violations,
        )

    # ========================================================
    # 7. Verify training manifest.
    # ========================================================

    if "training_manifest.json" in files:

        verify_training_manifest(
            manifest,
            files,
            violations,
        )

    # ========================================================
    # 8. Verify evaluation.
    # ========================================================

    if (
        "evaluation.json" in files
        and "training_manifest.json" in files
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

    # ========================================================
    # 9. Verify model card.
    # ========================================================

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

                positions = (
                    find_model_cards(
                        readme
                    )
                )

                if len(positions) == 0:

                    add_violation(
                        violations,
                        "MODEL_CARD_COUNT",
                    )

                    add_violation(
                        violations,
                        "MISSING_MODEL_CARD",
                    )

                elif len(positions) > 1:

                    add_violation(
                        violations,
                        "MODEL_CARD_COUNT",
                    )

                else:

                    add_violation(
                        violations,
                        "MODEL_CARD_MISMATCH",
                    )

    # ========================================================
    # 10. Deduplicate and sort violations.
    # ========================================================

    sorted_violations = sorted(
        violations,
        key=lambda x: x.encode("utf-8"),
    )

    # ========================================================
    # 11. Decide admit/reject.
    # ========================================================

    if len(sorted_violations) == 0:
        decision = "admit"
    else:
        decision = "reject"

    # ========================================================
    # 12. EXACT response shape.
    # ========================================================

    return jsonify({
        "decision": decision,
        "violations": sorted_violations,
        "inventoryDigest": inventory_digest,
    })


# ============================================================
# LOCAL / RENDER STARTUP
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
